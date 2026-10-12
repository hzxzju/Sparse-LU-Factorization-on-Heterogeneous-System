#include <iostream>
#include <vector>
#include <set>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>
#include "symbolic.h"
#include "numeric.h"
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
}

int main(int argc, char** argv)
{
    Timer t, total_timer, symbolic_timer;
    double preprocess_ms = 0., symbolic_ms = 0., csr_ms = 0.;
    double predict_ms = 0., leveling_ms = 0., symbolic_total_ms = 0.;
    double numeric_ms = 0., solve_ms = 0., total_ms = 0.;
    SNicsLU *nicslu = nullptr;

    const char *matrixName = nullptr;
    bool PERTURB = false;
    UpdateStrategy update_strategy = UpdateStrategy::RightLooking;
    SchedulingStrategy scheduling = SchedulingStrategy::Level;

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
    } catch (const std::invalid_argument &e) {
        cerr << e.what() << endl;
        return -1;
    }

    // Task 1 S2: T_gpu is the complete heterogeneous solve wall time,
    // including input preparation, CPU work, transfers and GPU synchronization.
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

    // Buffer solver diagnostics so terminal output does not delay the solve.
    std::ostringstream solver_output;
    symbolic_timer.start();
    t.start();

    // B's CSR arrays are also CSC arrays of B^T. Analyze and factor B^T;
    // solve_CSR below uses the transpose factors to solve the original system.
    Symbolic_Matrix A_sym(n, solver_output, cerr);
    A_sym.fill_in(ai, ap);
    t.elapsedWallclockTime(symbolic_ms);

    t.start();
    A_sym.csr();
    t.elapsedWallclockTime(csr_ms);

    t.start();
    A_sym.predictLU(ai, ap, ax);
    t.elapsedWallclockTime(predict_ms);

    if (scheduling == SchedulingStrategy::Level) {
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
    t.start();
    LUonDevice(A_sym, solver_output, numeric_errors, PERTURB, update_strategy, scheduling);
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
    t.start();
    vector<REAL> x = A_sym.solve_CSR(nicslu, b);
    t.elapsedWallclockTime(solve_ms);
    // Factors have been copied back by blocking hcMemcpy inside LUonDevice;
    // solve_CSR is synchronous CPU work. The result is now available on host.
    total_timer.elapsedWallclockTime(total_ms);

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
        << "Task 1 total solve time (T_gpu): " << total_ms << " ms\n";
    cout << timing_output.str();

    // Output serialization and residual verification are outside T_gpu.
    {
        ofstream x_f("x.dat");
        x_f << std::fixed << std::setprecision(10);
        for (REAL xx: x)
            x_f << xx << '\n';
    }

    // Residual verification: compute ||Ax - b|| using the original matrix
    {
        REAL n1 = 0., n2 = 0., ni = 0.;
        for (unsigned i = 0; i < n; ++i) {
            REAL sum = 0.;
            unsigned end = ap_backup[i + 1];
            for (unsigned j = ap_backup[i]; j < end; ++j) {
                sum += (REAL)ax_backup[j] * x[ai_backup[j]];
            }
            REAL tmp = sum - b[i];
            if (tmp < 0.) tmp = -tmp;
            n1 += tmp;
            n2 += tmp * tmp;
            if (tmp > ni) ni = tmp;
        }
        cout << "Ax-b (1-norm): " << n1 << endl;
        cout << "Ax-b (2-norm): " << sqrt(n2) << endl;
        cout << "Ax-b (infinite-norm): " << ni << endl;
    }

    NicsLU_Destroy(nicslu);
    free(nicslu);
    free(ax);
    free(ai);
    free(ap);
    free(ax_backup);
    free(ai_backup);
    free(ap_backup);

    return 0;
}
