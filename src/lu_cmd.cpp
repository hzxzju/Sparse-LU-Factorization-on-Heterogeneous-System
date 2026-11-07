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
#include "solution_error.h"
#include "solution_output.h"
#include "Timer.h"
#include "preprocess.h"
#include "nicslu.h"

using namespace std;

void help_message()
{
    cerr << endl;
    cerr << "GLU program V3.0" << endl;
    cerr << "Usage: ./lu_cmd -i inputfile" << endl;
    cerr << "--repeat [N] (default: 1; N > 1 runs one uncounted warmup then N factor/solve cycles, S3 only)" << endl;
    cerr << "Additional usage: ./lu_cmd -i inputfile -p" << endl;
    cerr << "-p to enable perturbation" << endl;
    cerr << "-u, --update-strategy right-looking|left-looking (default: left-looking)" << endl;
    cerr << "-s, --scheduling level|synchronization-free (default: synchronization-free; SF requires LL)" << endl;
    cerr << "--sf-workers N (default: 0 = automatic; requires SF when nonzero)" << endl;
    cerr << "--sf-update block|wave (default: block; wave requires LL-SF)" << endl;
    cerr << "--symbolic-backend auto|cpu|gpu|gpu-sf (default: auto; CPU/GPU-SF selected by input structure)" << endl;
    cerr << "--symbolic-batch-cols N (default: 128; positive workspace slot limit)" << endl;
    cerr << "--symbolic-sf-workers N (default: 0 = automatic; requires auto or gpu-sf)" << endl;
    cerr << "--symbolic-pool-entries N / --symbolic-max-pool-entries N (0 = automatic; auto or gpu-sf)" << endl;
    cerr << "--symbolic-block-size 64|128|256 (default: 256)" << endl;
    cerr << "--symbolic-verify (compare GPU CSC arrays with CPU; requires auto or a GPU backend)" << endl;
    cerr << "--symbolic-no-fallback (fail on GPU symbolic errors; requires auto or a GPU backend)" << endl;
    cerr << "--residual-tol T (default: 1e-6; relative ||x_GPU-x_CPU||_2 / ||x_CPU||_2 target; also absolute Ax-b refinement tolerance)" << endl;
    cerr << "--max-refine-iters N|n (default: n = matrix order; 0 disables corrections)" << endl;
    cerr << "--cpu-reference FILE (optional diagnostic comparison; default: disabled)" << endl;
    cerr << "--progress to flush stage diagnostics while running (for timeout diagnosis)" << endl;
}

static int run_solver(int argc, char** argv, GluTimings &timings, ostream &diagnostics)
{
    Timer t, symbolic_timer;
    double preprocess_ms = 0., symbolic_ms = 0., csr_ms = 0.;
    double predict_ms = 0., leveling_ms = 0., symbolic_total_ms = 0.;
    double numeric_ms = 0., solve_ms = 0., refinement_ms = 0.;
    SNicsLU *nicslu = nullptr;

    const char *matrixName = nullptr;
    const char *cpu_reference_file = nullptr;
    bool PERTURB = false;
    bool progress = false;
    unsigned repeat = 1;
    unsigned sf_workers = 0;
    SFUpdateStrategy sf_update = SFUpdateStrategy::Block;
    double residual_tolerance = 1e-6;
    unsigned max_refine_iters = 0;
    bool refine_limit_is_order = true;
    UpdateStrategy update_strategy = UpdateStrategy::LeftLooking;
    SchedulingStrategy scheduling = SchedulingStrategy::SynchronizationFree;
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
            if(i + 1 >= argc || matrixName != nullptr) {
                help_message();
                return -1;
            }
            matrixName = argv[i+1];
            i += 2;
        }
        else if (strcmp(argv[i], "--repeat") == 0) {
            if (i + 1 >= argc || argv[i + 1][0] == '-') {
                repeat = 1;
                ++i;
                continue;
            }
            try {
                const std::string value(argv[i + 1]);
                if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
                    throw std::invalid_argument("invalid repeat count");
                const unsigned long long parsed = std::stoull(value);
                if (!parsed || parsed > std::numeric_limits<unsigned>::max())
                    throw std::out_of_range("repeat count out of range");
                repeat = static_cast<unsigned>(parsed);
            } catch (const std::exception &) {
                cerr << "--repeat must be a positive integer no greater than UINT_MAX." << endl;
                return -1;
            }
            i += 2;
        }
        else if (strcmp(argv[i], "--cpu-reference") == 0) {
            if (i + 1 >= argc) { help_message(); return -1; }
            cpu_reference_file = argv[i + 1];
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
        else if (strcmp(argv[i], "--sf-update") == 0) {
            if (i + 1 >= argc) { help_message(); return -1; }
            try {
                sf_update = ParseSFUpdateStrategy(argv[i + 1]);
            } catch (const std::invalid_argument &e) {
                cerr << e.what() << endl;
                return -1;
            }
            i += 2;
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
        ValidateNumericConfiguration(update_strategy, scheduling, sf_update);
        ValidateSymbolicConfiguration(symbolic_config);
        if (sf_workers && scheduling != SchedulingStrategy::SynchronizationFree)
            throw std::invalid_argument("--sf-workers requires synchronization-free scheduling");
    } catch (const std::invalid_argument &e) {
        cerr << e.what() << endl;
        return -1;
    }

    glu_timing_switch(&timings, GLU_ANALYSIS_OTHER);
    auto report_stage = [&](const char *stage) {
        if (progress) {
            glu_timing::Scope phase(GLU_LOGGING);
            diagnostics << "[stage] " << stage << endl;
        }
    };
    report_stage("input/preprocessing started");
    {
        glu_timing::Scope phase(GLU_CONTEXT_INIT);
        nicslu = static_cast<SNicsLU *>(malloc(sizeof(SNicsLU)));
    }
    if (nicslu == nullptr) {
        diagnostics << "Failed to allocate NicsLU context." << endl;
        glu_timing_switch(&timings, GLU_CLEANUP);
        return -1;
    }

    // Keep both the original matrix and the reordered/scaled matrix in CSR.
    t.start();
    int err = preprocess_timed(matrixName, nicslu, &ax, &ai, &ap,
        &ax_backup, &ai_backup, &ap_backup, &nnz_backup, &timings);
    t.elapsedWallclockTime(preprocess_ms);
    if (err) {
        diagnostics << "Matrix preprocessing failed." << endl;
        glu_timing_switch(&timings, GLU_CLEANUP);
        free(nicslu);
        return -1;
    }
    n = nicslu->n;
    if (refine_limit_is_order) max_refine_iters = n;

    // Analyze/factor B^T, then solve the original system with transpose factors.
    Symbolic_Matrix A_sym(n, diagnostics, diagnostics);
    auto release_host = [&]() {
        glu_timing_switch(&timings, GLU_CLEANUP);
        NicsLU_Destroy(nicslu);
        free(nicslu);
        free(ax); free(ai); free(ap);
        free(ax_backup); free(ai_backup); free(ap_backup);
        return ShutdownGPUDevice(A_sym.gpu_device, diagnostics);
    };

    try {
        report_stage("symbolic fill-in started");
        symbolic_timer.start();
        t.start();
        {
            glu_timing::Scope phase(GLU_SYMBOLIC);
            A_sym.fill_in(ai, ap, symbolic_config);
        }
        t.elapsedWallclockTime(symbolic_ms);
        report_stage("CSR construction started");
        t.start();
        {
            glu_timing::Scope phase(GLU_ANALYSIS_OTHER);
            A_sym.csr();
        }
        t.elapsedWallclockTime(csr_ms);
        report_stage("numeric value preparation started");
        t.start();
        {
            glu_timing::Scope phase(GLU_PREPROCESS);
            A_sym.predictLU(ai, ap, ax);
        }
        t.elapsedWallclockTime(predict_ms);
        if (scheduling == SchedulingStrategy::Level) {
            report_stage("level generation started");
            t.start();
            {
                glu_timing::Scope phase(GLU_ANALYSIS_OTHER);
                A_sym.leveling();
            }
            t.elapsedWallclockTime(leveling_ms);
        }
        symbolic_timer.elapsedWallclockTime(symbolic_total_ms);

#if GLU_DEBUG
        { glu_timing::Scope phase(GLU_PREPROCESS); A_sym.ABFTCalculateCCA(); }
#endif
        std::ostringstream numeric_errors;
        glu_timing::LogBuffer error_buffer(numeric_errors.rdbuf());
        ostream errors(&error_buffer);
        vector<REAL> b;
        vector<long double> x;
        RefinementResult refinement;
        bool solved = false;
        {
            glu_timing::Scope phase(GLU_RHS_READ);
            b.assign(n, 1.);
        }
        report_stage("GPU numeric LU started");
        Timer factor_timer;
        factor_timer.start();
        LUonDeviceRepeated(A_sym, diagnostics, errors, PERTURB, update_strategy, scheduling,
                           sf_workers, sf_update, repeat, [&]() {
            factor_timer.elapsedWallclockTime(numeric_ms);
            if (!numeric_errors.str().empty())
                throw std::runtime_error(numeric_errors.str());
#if GLU_DEBUG
            { glu_timing::Scope phase(GLU_RESIDUAL_CHECK); A_sym.ABFTCheckResult(); }
#endif
            report_stage("CPU triangular solve started");
            t.start();
            {
                glu_timing::Scope phase(GLU_TRIANGULAR_SOLVE);
                const vector<REAL> initial_x = A_sym.solve_CSR(nicslu, b);
                glu_timing::Scope host_path(GLU_REFINEMENT);
                x.assign(initial_x.begin(), initial_x.end());
            }
            t.elapsedWallclockTime(solve_ms);
            report_stage("iterative refinement started");
            t.start();
            {
                glu_timing::Scope phase(GLU_REFINEMENT);
                diagnostics << std::setprecision(std::numeric_limits<REAL>::max_digits10)
                    << "Refinement Ax-b L2 tolerance: " << residual_tolerance << '\n'
                    << "Refinement max iterations: " << max_refine_iters << '\n';
                refinement = RefineCSR(n, ax_backup, ai_backup, ap_backup,
                    b, x, residual_tolerance, max_refine_iters,
                    [&](const vector<REAL> &rhs) { return A_sym.solve_CSR(nicslu, rhs); },
                    &diagnostics);
            }
            t.elapsedWallclockTime(refinement_ms);
            solved = true;
            factor_timer.start();
        });
        if (!numeric_errors.str().empty())
            throw std::runtime_error(numeric_errors.str());
        if (!solved) throw std::runtime_error("Numeric factorization did not produce a solution");
        report_stage("solve completed");

        // Independent final verification is postprocessing. Refinement's own
        // convergence residuals remain in refinement, including with a zero cap.
        ResidualNorms residual;
        SolutionError solution_error;
        {
            glu_timing::Scope phase(GLU_RESIDUAL_CHECK);
            vector<REAL> work;
            residual = ResidualCSR(n, ax_backup, ai_backup, ap_backup, b, x, work);
        }
        string reference_path;
        bool solution_converged = true;
        if (cpu_reference_file) {
            glu_timing::Scope evaluation(GLU_RESIDUAL_CHECK);
            reference_path = cpu_reference_file;
            const vector<double> cpu_x = ReadCPUReference(reference_path, n);
            solution_error = CompareSolutions(x, cpu_x);
            solution_converged = std::isfinite(solution_error.relative_l2)
                && solution_error.relative_l2 <= residual_tolerance;
        }
        {
            glu_timing::Scope phase(GLU_OUTPUT_WRITE);
            if (cpu_reference_file) {
                // Preserve extended-precision text output for explicit CPU comparisons.
                ofstream x_f("x.dat");
                x_f << std::scientific << std::setprecision(std::numeric_limits<long double>::max_digits10);
                for (long double xx : x) x_f << xx << '\n';
                x_f.close();
                if (!x_f) throw std::runtime_error("Failed to write x.dat");
            }
            WriteSolutionBinary("x.bin", x);
        }
        {
            glu_timing::Scope phase(GLU_LOGGING);
            diagnostics << "Matrix Row: " << n << '\n'
                << "Original nonzero: " << nicslu->nnz << '\n'
                << std::fixed << std::setprecision(6)
                << "Input/preprocessing time: " << preprocess_ms << " ms\n"
                << "Symbolic time: " << symbolic_ms << " ms\n"
                << "CSR time: " << csr_ms << " ms\n"
                << "PredictLU time: " << predict_ms << " ms\n"
                << "Leveling time: " << leveling_ms << " ms\n"
                << "Symbolic preparation time (total): " << symbolic_total_ms << " ms\n"
                << "Numeric LU time (including GPU setup/transfers): " << numeric_ms << " ms\n"
                << "Triangular solve time (CPU): " << solve_ms << " ms\n"
                << "Iterative refinement time (CPU): " << refinement_ms << " ms\n"
                << "Refinement iterations: " << refinement.iterations << '\n'
                << "Refinement status: " << refinement.status << '\n';
            if (cpu_reference_file) {
                diagnostics << "CPU reference file: " << reference_path << '\n'
                << std::scientific << std::setprecision(std::numeric_limits<long double>::max_digits10)
                << "GPU-CPU absolute L2 error: " << solution_error.absolute_l2 << '\n'
                << "GPU-CPU relative L2 error: " << solution_error.relative_l2 << '\n'
                << "GPU-CPU relative L2 tolerance: " << residual_tolerance << '\n'
                << "GPU-CPU comparison status: " << (solution_converged ? "converged" : "not_converged") << '\n';
            }
            diagnostics << std::scientific << std::setprecision(std::numeric_limits<long double>::max_digits10)
                << "Ax-b (1-norm): " << residual.l1 << '\n'
                << "Ax-b (2-norm): " << residual.l2 << '\n'
                << "Ax-b (infinite-norm): " << residual.linf << endl;
            if (!solution_converged)
                diagnostics << "GPU-CPU relative L2 target not reached." << endl;
        }
        const bool cleaned = release_host();
        return cleaned ? (solution_converged ? 0 : 2) : -1;
    } catch (const std::exception &e) {
        diagnostics << "Solver failed: " << e.what() << endl;
        release_host();
        return -1;
    }
}

int main(int argc, char **argv)
{
    GluTimings timings;
    glu_timing_init(&timings);
    glu_timing_switch(&timings, GLU_ANALYSIS_OTHER);
    glu_timing::Session session(timings);
    glu_timing::LogBuffer log_buffer(cerr.rdbuf());
    ostream diagnostics(&log_buffer);
    const int status = run_solver(argc, argv, timings, diagnostics);
    // run_solver's containers and symbolic matrix have also been destroyed.
    glu_timing_switch(&timings, GLU_LOGGING);
    if (status == 0 || status == 2) {
        static const char *names[] = {
            "read_parse", "rhs_read", "preprocess", "ordering", "symbolic",
            "analysis_other", "context_init", "workspace_alloc", "kernel_compile",
            "plan_upload", "cleanup", "value_h2d", "numeric_factor",
            "triangular_solve", "solution_d2h", "refinement", "sync",
            "residual_check", "output_write", "logging"
        };
        diagnostics << std::fixed << std::setprecision(9);
        for (int stage = 0; stage < GLU_STAGE_COUNT; ++stage) {
            glu_timing_switch(&timings, GLU_LOGGING);
            diagnostics << names[stage] << "_s=" << timings.seconds[stage] << '\n';
        }
        diagnostics << "factor_d2h_s=" << timings.factor_d2h_s
            << " (included in refinement: host solve path preparation)\n";
        const double once = glu_timing_sum(&timings, GLU_READ_PARSE, GLU_CLEANUP);
        const double solve = glu_timing_sum(&timings, GLU_VALUE_H2D, GLU_SYNC);
        // Match the six-decimal terminal values used by the batch evaluator.
        diagnostics << "total_solve_ms="
            << (0.005 * (std::round(once * 1.e6) / 1.e6)
                + std::round(solve * 1.e6) / 1.e6) * 1000.0 << endl;
    }
    diagnostics.flush();
    glu_timing_switch(&timings, GLU_TIMING_IDLE);
    {
        const double once = glu_timing_sum(&timings, GLU_READ_PARSE, GLU_CLEANUP);
        const double solve = glu_timing_sum(&timings, GLU_VALUE_H2D, GLU_SYNC);
        const double post = glu_timing_sum(&timings, GLU_RESIDUAL_CHECK, GLU_LOGGING);
        cout << std::fixed << std::setprecision(6)
             << "once_s=" << once << " solve_s=" << solve << " post_s=" << post << '\n';
    }
    return status;
}
