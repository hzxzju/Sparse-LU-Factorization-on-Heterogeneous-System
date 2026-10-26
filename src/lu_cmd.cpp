#include <iostream>
#include <vector>
#include <set>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>
#include <limits>
#include "symbolic.h"
#include "numeric.h"
#include "refinement.h"
#include "Timer.h"
#include "preprocess.h"
#include "nicslu.h"

using namespace std;

void help_message()
{
    cout << endl;
    cout << "GLU program V3.0" << endl;
    cout << "Usage: ./lu_cmd -i inputfile" << endl;
    cout << "Additional usage: ./lu_cmd -i inputfile -p" << endl;
    cout << "-p to enable perturbation" << endl;
    cout << "-u, --update-strategy right-looking|left-looking (default: right-looking)" << endl;
    cout << "-s, --scheduling level|synchronization-free (default: level; SF requires LL)" << endl;
    cout << "--sf-workers N (default: 0 = automatic; requires SF when nonzero)" << endl;
    cout << "--symbolic-backend cpu|gpu|gpu-sf (default: cpu; GPU fill-in, CPU CSR/values/levels)" << endl;
    cout << "--symbolic-batch-cols N (default: 128; positive workspace slot limit)" << endl;
    cout << "--symbolic-sf-workers N (default: 0 = automatic; requires gpu-sf)" << endl;
    cout << "--symbolic-pool-entries N / --symbolic-max-pool-entries N (0 = automatic; gpu-sf)" << endl;
    cout << "--symbolic-block-size 64|128|256 (default: 256)" << endl;
    cout << "--symbolic-verify (compare GPU CSC arrays with CPU; requires a GPU backend)" << endl;
    cout << "--symbolic-no-fallback (fail on GPU symbolic errors; requires a GPU backend)" << endl;
    cout << "--residual-tol T (default: 1e-6; absolute ||Ax-b||_2 target)" << endl;
    cout << "--max-refine-iters N|n (default: n = matrix order; 0 disables corrections)" << endl;
    cout << "--progress to flush stage diagnostics while running (for timeout diagnosis)" << endl;
}

int main(int argc, char** argv)
{
    Timer t, total_timer, symbolic_timer;
    double preprocess_ms = 0., symbolic_ms = 0., csr_ms = 0.;
    double predict_ms = 0., leveling_ms = 0., symbolic_total_ms = 0.;
    double numeric_ms = 0., solve_ms = 0., refinement_ms = 0., total_ms = 0.;
    SNicsLU *nicslu = nullptr;

    const char *matrixName = nullptr;
    bool PERTURB = false;
    bool progress = false;
    unsigned sf_workers = 0;
    double residual_tolerance = 1e-6;
    unsigned max_refine_iters = 0;
    bool refine_limit_is_order = true;
    UpdateStrategy update_strategy = UpdateStrategy::RightLooking;
    SchedulingStrategy scheduling = SchedulingStrategy::Level;
    SymbolicConfig symbolic_config;

    double *ax = NULL, *ax_backup = NULL;
    unsigned int *ai = NULL, *ai_backup = NULL;
    unsigned int *ap = NULL, *ap_backup = NULL;
    unsigned int n, nnz_backup;

    if (argc < 3) {
        help_message();
        return -1;
    }

    for (int i = 1; i < argc;) {
        if (strcmp(argv[i], "-i") == 0) {
            if(i + 1 >= argc) {
                help_message();
                return -1;
            }
            matrixName = argv[i+1];
            i += 2;
        }
        else if (strcmp(argv[i], "-p") == 0) {
            PERTURB = true;
            i += 1;
        }
        else if (strcmp(argv[i], "--progress") == 0) {
            progress = true;
            i += 1;
        }
        else if (strcmp(argv[i], "--symbolic-backend") == 0) {
            if (i + 1 >= argc) { help_message(); return -1; }
            try {
                symbolic_config.backend = ParseSymbolicBackend(argv[i + 1]);
            } catch (const std::invalid_argument &e) {
                cerr << e.what() << endl;
                return -1;
            }
            i += 2;
        }
        else if (strcmp(argv[i], "--symbolic-batch-cols") == 0 ||
                 strcmp(argv[i], "--symbolic-block-size") == 0 ||
                 strcmp(argv[i], "--symbolic-sf-workers") == 0 ||
                 strcmp(argv[i], "--symbolic-pool-entries") == 0 ||
                 strcmp(argv[i], "--symbolic-max-pool-entries") == 0) {
            if (i + 1 >= argc) { help_message(); return -1; }
            try {
                const std::string value(argv[i + 1]);
                if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
                    throw std::invalid_argument("invalid symbolic integer");
                const unsigned long long parsed = std::stoull(value);
                if ((!parsed && (strcmp(argv[i], "--symbolic-batch-cols") == 0 ||
                                 strcmp(argv[i], "--symbolic-block-size") == 0)) ||
                    parsed > std::numeric_limits<unsigned>::max())
                    throw std::out_of_range("symbolic integer out of range");
                if (strcmp(argv[i], "--symbolic-batch-cols") == 0)
                    symbolic_config.batch_cols = static_cast<unsigned>(parsed);
                else if (strcmp(argv[i], "--symbolic-block-size") == 0)
                    symbolic_config.block_size = static_cast<unsigned>(parsed);
                else if (strcmp(argv[i], "--symbolic-sf-workers") == 0)
                    symbolic_config.sf_workers = static_cast<unsigned>(parsed);
                else if (strcmp(argv[i], "--symbolic-pool-entries") == 0)
                    symbolic_config.pool_entries = static_cast<unsigned>(parsed);
                else
                    symbolic_config.max_pool_entries = static_cast<unsigned>(parsed);
            } catch (const std::exception &) {
                cerr << argv[i] << " must be an unsigned integer (batch/block size must be positive)." << endl;
                return -1;
            }
            i += 2;
        }
        else if (strcmp(argv[i], "--symbolic-verify") == 0) {
            symbolic_config.verify = true;
            i += 1;
        }
        else if (strcmp(argv[i], "--symbolic-no-fallback") == 0) {
            symbolic_config.fallback_to_cpu = false;
            i += 1;
        }
        else if (strcmp(argv[i], "--sf-workers") == 0) {
            if (i + 1 >= argc) { help_message(); return -1; }
            const std::string value(argv[i + 1]);
            try {
                if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
                    throw std::invalid_argument("invalid worker count");
                const unsigned long long parsed = std::stoull(value);
                if (parsed > std::numeric_limits<unsigned>::max())
                    throw std::out_of_range("worker count too large");
                sf_workers = static_cast<unsigned>(parsed);
            } catch (const std::exception &) {
                cerr << "--sf-workers must be an unsigned integer (0 = automatic)." << endl;
                return -1;
            }
            i += 2;
        }
        else if (strcmp(argv[i], "--residual-tol") == 0) {
            if (i + 1 >= argc) { help_message(); return -1; }
            try {
                const std::string value(argv[i + 1]);
                size_t consumed = 0;
                residual_tolerance = std::stod(value, &consumed);
                if (consumed != value.size() || !std::isfinite(residual_tolerance) ||
                    residual_tolerance <= 0)
                    throw std::invalid_argument("invalid residual tolerance");
            } catch (const std::exception &) {
                cerr << "--residual-tol must be finite and positive." << endl;
                return -1;
            }
            i += 2;
        }
        else if (strcmp(argv[i], "--max-refine-iters") == 0) {
            if (i + 1 >= argc) { help_message(); return -1; }
            const std::string value(argv[i + 1]);
            refine_limit_is_order = value == "n";
            if (!refine_limit_is_order) {
                try {
                    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
                        throw std::invalid_argument("invalid refinement limit");
                    const unsigned long long parsed = std::stoull(value);
                    if (parsed > std::numeric_limits<unsigned>::max())
                        throw std::out_of_range("refinement limit too large");
                    max_refine_iters = static_cast<unsigned>(parsed);
                } catch (const std::exception &) {
                    cerr << "--max-refine-iters must be an unsigned integer or n." << endl;
                    return -1;
                }
            }
            i += 2;
        }
        else if (strcmp(argv[i], "-u") == 0 || strcmp(argv[i], "--update-strategy") == 0) {
            if (i + 1 >= argc) {
                help_message();
                return -1;
            }
            try {
                update_strategy = ParseUpdateStrategy(argv[i + 1]);
            } catch (const std::invalid_argument &e) {
                cerr << e.what() << endl;
                return -1;
            }
            i += 2;
        }
        else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--scheduling") == 0) {
            if (i + 1 >= argc) {
                help_message();
                return -1;
            }
            try {
                scheduling = ParseSchedulingStrategy(argv[i + 1]);
            } catch (const std::invalid_argument &e) {
                cerr << e.what() << endl;
                return -1;
            }
            i += 2;
        }
        else {
            help_message();
            return -1;
        }
    }

    if (matrixName == nullptr) {
        help_message();
        return -1;
    }

    try {
        ValidateNumericConfiguration(update_strategy, scheduling);
        ValidateSymbolicConfiguration(symbolic_config);
        if (sf_workers && scheduling != SchedulingStrategy::SynchronizationFree)
            throw std::invalid_argument("--sf-workers requires synchronization-free scheduling");
    } catch (const std::invalid_argument &e) {
        cerr << e.what() << endl;
        return -1;
    }

    // Task 1 S2: T_gpu is the complete heterogeneous solve wall time,
    // including input preparation, CPU work, transfers and GPU synchronization.
    auto report_stage = [&](const char *stage) {
        if (progress)
            cerr << "[stage] " << stage << endl;
    };
    report_stage("input/preprocessing started");
    total_timer.start();
    nicslu = static_cast<SNicsLU *>(malloc(sizeof(SNicsLU)));
    if (nicslu == nullptr) {
        cerr << "Failed to allocate NicsLU context." << endl;
        return -1;
    }

    // Keep the original matrix and the reordered/scaled matrix B in CSR.
    t.start();
    int err = preprocess(matrixName, nicslu, &ax, &ai, &ap, &ax_backup, &ai_backup, &ap_backup, &nnz_backup);
    if (err)
    {
        cerr << "Matrix preprocessing failed." << endl;
        free(nicslu);
        return -1;
    }
    t.elapsedWallclockTime(preprocess_ms);

    n = nicslu->n;
    if (refine_limit_is_order)
        max_refine_iters = n;

    // Keep benchmark diagnostics buffered unless live progress was requested.
    std::ostringstream solver_output;
    ostream &diagnostics = progress ? cout : solver_output;
    report_stage("symbolic fill-in started");
    symbolic_timer.start();
    t.start();

    // B's CSR arrays are also CSC arrays of B^T. Analyze and factor B^T;
    // solve_CSR below uses the transpose factors to solve the original system.
    Symbolic_Matrix A_sym(n, diagnostics, cerr);
    try {
        A_sym.fill_in(ai, ap, symbolic_config);
    } catch (const std::exception &e) {
        cout << solver_output.str();
        cerr << "Symbolic factorization failed: " << e.what() << endl;
        NicsLU_Destroy(nicslu);
        free(nicslu);
        free(ax); free(ai); free(ap);
        free(ax_backup); free(ai_backup); free(ap_backup);
        return -1;
    }
    t.elapsedWallclockTime(symbolic_ms);

    report_stage("CSR construction started");
    t.start();
    A_sym.csr();
    t.elapsedWallclockTime(csr_ms);

    report_stage("numeric value preparation started");
    t.start();
    A_sym.predictLU(ai, ap, ax);
    t.elapsedWallclockTime(predict_ms);

    if (scheduling == SchedulingStrategy::Level) {
        report_stage("level generation started");
        t.start();
        A_sym.leveling();
        t.elapsedWallclockTime(leveling_ms);
    }
    symbolic_timer.elapsedWallclockTime(symbolic_total_ms);

#if GLU_DEBUG
    A_sym.ABFTCalculateCCA();
//    A_sym.PrintLevel();
#endif

    // Numeric factorization on GPU updates A_sym.val in place to LU factors.
    std::ostringstream numeric_errors;
    report_stage("GPU numeric LU started");
    t.start();
    LUonDevice(A_sym, diagnostics, numeric_errors, PERTURB, update_strategy, scheduling, sf_workers);
    t.elapsedWallclockTime(numeric_ms);
    if (!numeric_errors.str().empty()) {
        cout << solver_output.str();
        cerr << numeric_errors.str();
        NicsLU_Destroy(nicslu);
        free(nicslu);
        free(ax); free(ai); free(ap);
        free(ax_backup); free(ai_backup); free(ap_backup);
        return -1;
    }

#if GLU_DEBUG
    A_sym.ABFTCheckResult();
#endif

    // Solve Ax=b with the computed factors and NICSLU permutations/scales.
    vector<REAL> b(n, 1.);
    report_stage("CPU triangular solve started");
    t.start();
    vector<REAL> x = A_sym.solve_CSR(nicslu, b);
    t.elapsedWallclockTime(solve_ms);
    report_stage("residual check / iterative refinement started");
    t.start();
    diagnostics << std::setprecision(std::numeric_limits<REAL>::max_digits10)
                << "Residual L2 tolerance: " << residual_tolerance << '\n'
                << "Refinement max iterations: " << max_refine_iters << '\n';
    const RefinementResult refinement = RefineCSR(n, ax_backup, ai_backup, ap_backup,
        b, x, residual_tolerance, max_refine_iters,
        [&](const vector<REAL> &rhs) { return A_sym.solve_CSR(nicslu, rhs); },
        &diagnostics);
    t.elapsedWallclockTime(refinement_ms);
    // Include the initial residual check and every correction in T_gpu.
    total_timer.elapsedWallclockTime(total_ms);
    report_stage("solve completed");

    cout << "Matrix Row: " << n << endl;
    cout << "Original nonzero: " << nicslu->nnz << endl;
    cout << solver_output.str();
    std::ostringstream timing_output;
    timing_output << std::fixed << std::setprecision(6)
        << "Input/preprocessing time: " << preprocess_ms << " ms\n"
        << "Symbolic time: " << symbolic_ms << " ms\n"
        << "CSR time: " << csr_ms << " ms\n"
        << "PredictLU time: " << predict_ms << " ms\n"
        << "Leveling time: " << leveling_ms << " ms\n"
        << "Symbolic preparation time (total): " << symbolic_total_ms << " ms\n"
        << "Numeric LU time (including GPU setup/transfers): " << numeric_ms << " ms\n"
        << "Triangular solve time (CPU): " << solve_ms << " ms\n"
        << "Iterative refinement time (CPU): " << refinement_ms << " ms\n"
        << "Task 1 total solve time (T_gpu): " << total_ms << " ms\n";
    cout << timing_output.str();

    cout << "Refinement iterations: " << refinement.iterations << '\n'
         << "Refinement status: " << refinement.status << '\n';

    // Serialization and reporting are outside T_gpu; residual evaluation is inside.
    {
        ofstream x_f("x.dat");
        x_f << std::scientific << std::setprecision(std::numeric_limits<REAL>::max_digits10);
        for (REAL xx: x)
            x_f << xx << '\n';
    }

    cout << std::setprecision(std::numeric_limits<REAL>::max_digits10)
         << "Ax-b (1-norm): " << refinement.residual.l1 << '\n'
         << "Ax-b (2-norm): " << refinement.residual.l2 << '\n'
         << "Ax-b (infinite-norm): " << refinement.residual.linf << endl;
    if (!refinement.converged)
        cerr << "Residual L2 target not reached (" << refinement.status << ")." << endl;

    NicsLU_Destroy(nicslu);
    free(nicslu);
    free(ax);
    free(ai);
    free(ap);
    free(ax_backup);
    free(ai_backup);
    free(ap_backup);

    return refinement.converged ? 0 : 2;
}
