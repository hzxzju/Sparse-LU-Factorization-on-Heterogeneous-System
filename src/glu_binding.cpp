#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nicslu.h"
#include "preprocess.h"
#include "symbolic.h"
#include "numeric.h"
#include "type.h"

namespace py = pybind11;

class GLUFactorization {
public:
    GLUFactorization(
        py::array_t<double,  py::array::c_style> data,
        py::array_t<int32_t, py::array::c_style> indices,
        py::array_t<int32_t, py::array::c_style> indptr,
        int n_dim,
        bool perturb,
        const std::string &update_strategy,
        const std::string &scheduling,
        unsigned sf_workers,
        const std::string &symbolic_backend,
        unsigned symbolic_batch_cols,
        unsigned symbolic_block_size,
        bool symbolic_verify,
        bool symbolic_no_fallback,
        unsigned symbolic_sf_workers,
        unsigned symbolic_pool_entries,
        unsigned symbolic_max_pool_entries);

    GLUFactorization(const GLUFactorization &) = delete;
    GLUFactorization &operator=(const GLUFactorization &) = delete;
    GLUFactorization(GLUFactorization &&) = delete;
    GLUFactorization &operator=(GLUFactorization &&) = delete;

    py::array_t<double> solve(
        py::array_t<double, py::array::c_style> b);

    unsigned int n() const { return m_n; }
    const char *update_strategy() const { return UpdateStrategyName(m_update_strategy); }
    const char *scheduling() const { return SchedulingStrategyName(m_scheduling); }
    const char *symbolic_backend() const { return SymbolicBackendName(m_sym->symbolic_backend); }
    std::string diagnostics() const { return m_out.str(); }

private:
    struct NicsLUDeleter {
        void operator()(SNicsLU *matrix) const {
            if (matrix) { NicsLU_Destroy(matrix); std::free(matrix); }
        }
    };
    // Symbolic_Matrix retains stream references throughout solve() and reuse.
    // Declare streams first so they also outlive destruction of the matrix.
    std::ostringstream m_out, m_err;
    std::unique_ptr<SNicsLU, NicsLUDeleter> m_nicslu;
    std::unique_ptr<Symbolic_Matrix> m_sym;
    unsigned          m_n      = 0;
    UpdateStrategy    m_update_strategy = UpdateStrategy::RightLooking;
    SchedulingStrategy m_scheduling = SchedulingStrategy::Level;
};


GLUFactorization::GLUFactorization(
    py::array_t<double,  py::array::c_style> data,
    py::array_t<int32_t, py::array::c_style> indices,
    py::array_t<int32_t, py::array::c_style> indptr,
    int n_dim,
    bool perturb,
    const std::string &update_strategy,
    const std::string &scheduling,
    unsigned sf_workers,
    const std::string &symbolic_backend,
    unsigned symbolic_batch_cols,
    unsigned symbolic_block_size,
    bool symbolic_verify,
    bool symbolic_no_fallback,
    unsigned symbolic_sf_workers,
    unsigned symbolic_pool_entries,
    unsigned symbolic_max_pool_entries)
{
    m_update_strategy = ParseUpdateStrategy(update_strategy);
    m_scheduling = ParseSchedulingStrategy(scheduling);
    ValidateNumericConfiguration(m_update_strategy, m_scheduling);
    if (sf_workers && m_scheduling != SchedulingStrategy::SynchronizationFree)
        throw std::invalid_argument("sf_workers requires synchronization-free scheduling");
    SymbolicConfig symbolic_config;
    symbolic_config.backend = ParseSymbolicBackend(symbolic_backend);
    symbolic_config.batch_cols = symbolic_batch_cols;
    symbolic_config.block_size = symbolic_block_size;
    symbolic_config.verify = symbolic_verify;
    symbolic_config.fallback_to_cpu = !symbolic_no_fallback;
    symbolic_config.sf_workers = symbolic_sf_workers;
    symbolic_config.pool_entries = symbolic_pool_entries;
    symbolic_config.max_pool_entries = symbolic_max_pool_entries;
    ValidateSymbolicConfiguration(symbolic_config);
    if (n_dim <= 0)
        throw std::invalid_argument("n must be positive");
    if (data.ndim() != 1 || indices.ndim() != 1 || indptr.ndim() != 1)
        throw std::invalid_argument("CSC arrays must be one-dimensional");
    if (static_cast<size_t>(data.size()) > std::numeric_limits<unsigned>::max())
        throw std::invalid_argument("CSC nnz exceeds the unsigned index interface");

    unsigned n   = static_cast<unsigned>(n_dim);
    unsigned nnz = static_cast<unsigned>(data.shape(0));

    if (indices.shape(0) != static_cast<py::ssize_t>(nnz))
        throw std::invalid_argument("data and indices length mismatch");
    if (indptr.shape(0) != static_cast<py::ssize_t>(n + 1))
        throw std::invalid_argument("indptr length must be n+1");

    auto idx = indices.unchecked<1>();
    auto ptr = indptr.unchecked<1>();

    /* Convert int32 indices to unsigned int (validate non-negative) */
    std::vector<unsigned> ai_u(nnz), ap_u(n + 1);
    for (unsigned i = 0; i < nnz; ++i) {
        if (idx(i) < 0 || static_cast<unsigned>(idx(i)) >= n)
            throw std::invalid_argument("Row index out of range in CSC matrix");
        ai_u[i] = static_cast<unsigned>(idx(i));
    }
    for (unsigned i = 0; i <= n; ++i) {
        if (ptr(i) < 0)
            throw std::invalid_argument("Negative indptr value");
        ap_u[i] = static_cast<unsigned>(ptr(i));
        if (ap_u[i] > nnz || (i && ap_u[i] < ap_u[i - 1]))
            throw std::invalid_argument("CSC indptr must be monotone and bounded by nnz");
    }
    if (ap_u[0] != 0 || ap_u[n] != nnz)
        throw std::invalid_argument("CSC indptr must start at zero and end at nnz");

    // preprocess_from_arrays destroys NICSLU on failure; the temporary owner
    // frees only the outer struct until preprocessing succeeds.
    std::unique_ptr<SNicsLU, decltype(&std::free)> preprocessed(
        static_cast<SNicsLU *>(std::malloc(sizeof(SNicsLU))), &std::free);
    if (!preprocessed)
        throw std::bad_alloc();

    double       *ax_out = nullptr;
    unsigned int *ai_out = nullptr;
    unsigned int *ap_out = nullptr;

    int ret = preprocess_from_arrays(
        n, nnz,
        data.data(),
        ai_u.data(),
        ap_u.data(),
        preprocessed.get(),
        &ax_out, &ai_out, &ap_out);
    std::unique_ptr<double, decltype(&std::free)> ax_owner(ax_out, &std::free);
    std::unique_ptr<unsigned, decltype(&std::free)> ai_owner(ai_out, &std::free);
    std::unique_ptr<unsigned, decltype(&std::free)> ap_owner(ap_out, &std::free);

    if (ret != 0) {
        throw std::runtime_error(
            "GLU preprocessing failed (error " + std::to_string(ret) + ")");
    }
    m_nicslu.reset(preprocessed.release());

    m_n = m_nicslu->n;

    m_sym.reset(new Symbolic_Matrix(m_n, m_out, m_err));
    m_sym->fill_in(ai_out, ap_out, symbolic_config);
    m_sym->csr();
    m_sym->predictLU(ai_out, ap_out, ax_out);
    if (m_scheduling == SchedulingStrategy::Level)
        m_sym->leveling();

    /* CSC arrays no longer needed after symbolic phase */
    ax_owner.reset(); ai_owner.reset(); ap_owner.reset();

    /* GPU factorization — overwrites m_sym->val with LU factors */
    LUonDevice(*m_sym, m_out, m_err, perturb, m_update_strategy, m_scheduling, sf_workers);

    /* Surface device-factorization errors reported by the GPU backend. */
    std::string err_msg = m_err.str();
    if (!err_msg.empty()) {
        // A runtime/launch failure can leave finite, unfactorized values.
        // Surface all backend errors rather than returning an invalid solver.
        throw std::runtime_error("GPU factorization failed: " + err_msg);
    }
}


py::array_t<double> GLUFactorization::solve(
    py::array_t<double, py::array::c_style> b_arr)
{
    if (b_arr.ndim() != 1)
        throw std::invalid_argument("RHS must be one-dimensional");
    if (b_arr.shape(0) != static_cast<py::ssize_t>(m_n))
        throw std::invalid_argument(
            "RHS length " + std::to_string(b_arr.shape(0)) +
            " != matrix dimension " + std::to_string(m_n));

    auto b_buf = b_arr.unchecked<1>();

    /* Convert to the project's REAL type. */
    std::vector<REAL> rhs(m_n);
    for (unsigned i = 0; i < m_n; ++i)
        rhs[i] = static_cast<REAL>(b_buf(i));

    std::vector<REAL> x_f = m_sym->solve(m_nicslu.get(), rhs);

    /* Return float64 regardless of the project's REAL typedef. */
    py::array_t<double> x_out(static_cast<py::ssize_t>(m_n));
    auto x_buf = x_out.mutable_unchecked<1>();
    for (unsigned i = 0; i < m_n; ++i)
        x_buf(i) = static_cast<double>(x_f[i]);

    return x_out;
}


PYBIND11_MODULE(_pyglu, m) {
    m.doc() = "GLU GPU-accelerated sparse LU factorization — Python bindings";

    py::class_<GLUFactorization>(m, "GLUFactorization",
        R"(GPU LU factorization of a sparse matrix.

        Constructed by pyglu.splu(). Call .solve(b) to solve Ax=b.
        )")
        .def(py::init<
                py::array_t<double,  py::array::c_style>,
                py::array_t<int32_t, py::array::c_style>,
                py::array_t<int32_t, py::array::c_style>,
                int,
                bool,
                const std::string &,
                const std::string &,
                unsigned,
                const std::string &,
                unsigned, unsigned, bool, bool,
                unsigned, unsigned, unsigned>(),
            py::arg("data"),
            py::arg("indices"),
            py::arg("indptr"),
            py::arg("n"),
            py::arg("perturb") = false,
            py::arg("update_strategy") = "right-looking",
            py::arg("scheduling") = "level",
            py::kw_only(),
            py::arg("sf_workers") = 0,
            py::arg("symbolic_backend") = "cpu",
            py::arg("symbolic_batch_cols") = 128,
            py::arg("symbolic_block_size") = 256,
            py::arg("symbolic_verify") = false,
            py::arg("symbolic_no_fallback") = false,
            py::arg("symbolic_sf_workers") = 0,
            py::arg("symbolic_pool_entries") = 0,
            py::arg("symbolic_max_pool_entries") = 0,
            R"(Factorize a sparse matrix given in CSC format.

            Parameters
            ----------
            data : ndarray, float64, shape (nnz,)
            indices : ndarray, int32, shape (nnz,)   — row indices
            indptr  : ndarray, int32, shape (n+1,)   — column pointers
            n       : int                            — matrix dimension
            perturb : bool                           — enable GESP perturbation
            update_strategy : str                    — right-looking or left-looking
            scheduling : str                         -- level or synchronization-free (LL only)
            sf_workers : int                         -- numeric SF workers; 0 = automatic
            symbolic_backend : str                   -- cpu, gpu (graph), or gpu-sf
            symbolic_batch_cols : int                -- graph GPU workspace slots
            symbolic_block_size : int                -- 64, 128, or 256 threads
            symbolic_verify : bool                   -- compare all CSC arrays to CPU
            symbolic_no_fallback : bool              -- propagate GPU symbolic failures
            symbolic_sf_workers : int                -- symbolic SF workers; 0 = automatic
            symbolic_pool_entries : int              -- initial SF arena size; 0 = automatic
            symbolic_max_pool_entries : int          -- SF arena cap; 0 = memory budget
            )")
        .def("solve", &GLUFactorization::solve, py::arg("b"),
            R"(Solve Ax = b.

            Parameters
            ----------
            b : ndarray, float64, shape (n,)

            Returns
            -------
            x : ndarray, float64, shape (n,)
            )")
        .def_property_readonly("n", &GLUFactorization::n,
            "Matrix dimension.")
        .def_property_readonly("update_strategy", &GLUFactorization::update_strategy,
            "Canonical name of the numeric update strategy.")
        .def_property_readonly("scheduling", &GLUFactorization::scheduling,
            "Canonical name of the numeric scheduling strategy.")
        .def_property_readonly("symbolic_backend", &GLUFactorization::symbolic_backend,
            "Backend that produced the symbolic pattern (cpu after fallback).")
        .def_property_readonly("diagnostics", &GLUFactorization::diagnostics,
            "Symbolic timing, resource counters, fallback and numeric diagnostics.");
}
