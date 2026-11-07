/* klu_demo.c
 *
 * Minimal, self-contained KLU demonstration: read a sparse matrix from a
 * coordinate (triplet / Matrix Market) file, solve
 *
 *      A x = b,   with b = [1, 1, ..., 1]^T
 *
 * and write the solution vector x to a text file (one value per line,
 * full double precision).
 *
 * Usage:
 *      klu_demo <matrix.mtx> <out_x.txt> [-expand]
 *
 * Options:
 *      -expand   Honor the "%MatrixMarket ... symmetric" tag by mirroring the
 *                stored triangle to build the full matrix (standard Matrix
 *                Market semantics).  Off by default: the stored entries are
 *                used as-is.  The matrices shipped with GLU are already in the
 *                form the GLU solver consumes, so leave -expand OFF when you
 *                want to reproduce GLU's linear system exactly.
 *
 * Accepted input formats:
 *      - Standard Matrix Market header ("%%MatrixMarket ...") or the single-%
 *        variant used by some of these files ("%MatrixMarket ...").
 *      - Bare size line with no header at all (e.g. add32_csr.mtx):
 *              <nrows> <ncols> <nnz>
 *              i j v
 *              ...
 *      - 1-based indices, arbitrarily ordered, duplicate (i,j) entries are
 *        summed, blank lines and comment lines are ignored.
 *      - "symmetric" / "general" are recognized;  other fields are ignored.
 *
 * Build: see the Makefile in the parent directory (make).
 *
 * KLU API used (all default double/int32 precision):
 *      klu_defaults()        - fill in default parameters
 *      klu_analyze()         - symbolic factorization (fill-reducing ordering)
 *      klu_factor()          - numeric factorization
 *      klu_solve()           - solve using the factorization
 *      klu_free_symbolic()   - release the symbolic object
 *      klu_free_numeric()    - release the numeric object
 *
 * The common usage pattern for solving several right-hand sides is:
 *      analyze once, factor once, then call klu_solve() many times
 *      (optionally call klu_refactor() when only the values change).
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "klu.h"

static void die(const char *msg)
{
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(1);
}

static double wall_seconds(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) die("clock_gettime failed");
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

/* -------------------------------------------------------------------------- */
/* 1. Parse the coordinate file into triplet arrays (0-based, duplicates kept) */
/* -------------------------------------------------------------------------- */
int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <matrix.mtx> <out_x.txt> [-expand]\n", argv[0]);
        return 1;
    }

    int expand_symmetric = 0;                 /* default: use entries as-is */
    for (int a = 3; a < argc; ++a)
        if (strcmp(argv[a], "-expand") == 0) expand_symmetric = 1;

    FILE *f = fopen(argv[1], "r");
    if (!f) die("cannot open matrix file");

    char  line[8192];
    int   symmetric = 0;                       /* file says "symmetric" */
    long  n = -1, m = -1, nz = -1;

    /* Scan for the size line, skipping blank lines and comments.  Also look
     * for the "symmetric" keyword in any comment / header line. */
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '\n' || line[0] == '\r') continue;
        if (line[0] == '%') {
            if (strstr(line, "symmetric")) symmetric = 1;
            continue;
        }
        if (sscanf(line, "%ld %ld %ld", &n, &m, &nz) == 3) break;
        die("cannot parse size line");
    }
    if (n < 0 || nz < 0) die("missing size line");
    if (n != m)          die("matrix is not square");

    /* upper bound on entries after optional symmetric expansion */
    long cap = (symmetric && expand_symmetric) ? 2L * nz : nz;
    long *I = malloc(sizeof(long)   * (cap > 0 ? cap : 1));
    long *J = malloc(sizeof(long)   * (cap > 0 ? cap : 1));
    double *V = malloc(sizeof(double) * (cap > 0 ? cap : 1));
    if (!I || !J || !V) die("out of memory");

    long cnt = 0;
    while (cnt < nz && fgets(line, sizeof(line), f)) {
        if (line[0] == '%' || line[0] == '\n' || line[0] == '\r') continue;
        long i, j; double v;
        if (sscanf(line, "%ld %ld %lf", &i, &j, &v) != 3) continue; /* skip junk */
        if (i < 1 || i > n || j < 1 || j > n) die("index out of range");
        I[cnt] = i - 1; J[cnt] = j - 1; V[cnt] = v; cnt++;
        if (symmetric && expand_symmetric && i != j) {   /* mirror off-diagonal */
            I[cnt] = j - 1; J[cnt] = i - 1; V[cnt] = v; cnt++;
        }
    }
    fclose(f);
    if (cnt < nz) die("truncated matrix file");

    /* ---------------------------------------------------------------------- */
    /* 2. Convert triplets to CSC (compressed sparse column)                  */
    /*    KLU requires:                                                       */
    /*      - Ap[i] .. Ap[i+1]-1  are the entries of column i                 */
    /*      - Ai[] sorted ascending within each column                        */
    /*      - no duplicate row indices (they must be summed first)            */
    /* ---------------------------------------------------------------------- */
    long *colcount = calloc(n, sizeof(long));
    if (!colcount) die("out of memory");
    for (long k = 0; k < cnt; ++k) colcount[J[k]]++;

    int *Ap = malloc(sizeof(int)    * (n + 1));
    int *Ai = malloc(sizeof(int)    * (cnt > 0 ? cnt : 1));
    double *Ax = malloc(sizeof(double) * (cnt > 0 ? cnt : 1));
    long *pos  = malloc(sizeof(long) * n);
    if (!Ap || !Ai || !Ax || !pos) die("out of memory");

    Ap[0] = 0;
    for (long c = 0; c < n; ++c) Ap[c + 1] = Ap[c] + (int)colcount[c];
    for (long c = 0; c < n; ++c) pos[c] = Ap[c];
    for (long k = 0; k < cnt; ++k) {
        long c = J[k];
        Ai[pos[c]] = (int)I[k];
        Ax[pos[c]] = V[k];
        pos[c]++;
    }
    free(pos); free(I); free(J); free(V); free(colcount);

    /* insertion sort of row indices within each column (columns are short) */
    for (long c = 0; c < n; ++c) {
        long s = Ap[c], e = Ap[c + 1];
        for (long a = s + 1; a < e; ++a) {
            int    ri = Ai[a];
            double rv = Ax[a];
            long   b  = a;
            while (b > s && Ai[b - 1] > ri) { Ai[b] = Ai[b - 1]; Ax[b] = Ax[b - 1]; b--; }
            Ai[b] = ri; Ax[b] = rv;
        }
    }

    /* sum duplicates (now adjacent after the sort) */
    long w = 0;
    for (long c = 0; c < n; ++c) {
        long s = Ap[c], e = Ap[c + 1];
        long start = w;
        for (long k = s; k < e; ++k) {
            if (w > start && Ai[k] == Ai[w - 1]) Ax[w - 1] += Ax[k];
            else { Ai[w] = Ai[k]; Ax[w] = Ax[k]; w++; }
        }
        Ap[c] = (int)start;
    }
    Ap[n] = (int)w;
    long nnz_csc = w;

    /* ---------------------------------------------------------------------- */
    /* 3. KLU: symbolic analysis, numeric factorization, solve                */
    /* ---------------------------------------------------------------------- */
    klu_symbolic *Sym;
    klu_numeric  *Num;
    klu_common    Common;

    if (klu_defaults(&Common) != 1) die("klu_defaults failed");

    double started = wall_seconds();
    Sym = klu_analyze((int)n, Ap, Ai, &Common);
    double analyze_s = wall_seconds() - started;
    if (!Sym) { fprintf(stderr, "klu_analyze failed, status=%d\n", Common.status); return 2; }

    started = wall_seconds();
    Num = klu_factor(Ap, Ai, Ax, Sym, &Common);
    double factor_s = wall_seconds() - started;
    if (!Num) { fprintf(stderr, "klu_factor failed, status=%d\n", Common.status); return 2; }

    double *b = malloc(sizeof(double) * n);
    if (!b) die("out of memory");
    for (long i = 0; i < n; ++i) b[i] = 1.0;          /* right-hand side */

    /* klu_solve overwrites b with the solution x */
    started = wall_seconds();
    if (klu_solve(Sym, Num, (int)n, 1, b, &Common) != 1) {
        fprintf(stderr, "klu_solve failed, status=%d\n", Common.status);
        return 2;
    }
    double solve_s = wall_seconds() - started;
    fprintf(stderr, "analyze_s=%.17g factor_s=%.17g solve_s=%.17g total_s=%.17g\n",
            analyze_s, factor_s, solve_s, analyze_s + factor_s + solve_s);

    /* ---------------------------------------------------------------------- */
    /* 4. Residual ||A x - b||_2 on the matrix as assembled above             */
    /* ---------------------------------------------------------------------- */
    double *res = calloc(n, sizeof(double));
    if (!res) die("out of memory");
    for (long c = 0; c < n; ++c)
        for (long k = Ap[c]; k < Ap[c + 1]; ++k)
            res[Ai[k]] += Ax[k] * b[c];

    double n2 = 0.0, b2 = 0.0;
    for (long i = 0; i < n; ++i) {
        res[i] -= 1.0;
        n2 += res[i] * res[i];
        b2 += 1.0;
    }
    fprintf(stderr, "n=%ld nnz=%ld ||Ax-b||2=%.6e ||b||2=%.6e status=%d\n",
            n, nnz_csc, sqrt(n2), sqrt(b2), Common.status);

    /* ---------------------------------------------------------------------- */
    /* 5. Write x (full round-trip precision)                                 */
    /* ---------------------------------------------------------------------- */
    FILE *o = fopen(argv[2], "w");
    if (!o) die("cannot open output file");
    for (long i = 0; i < n; ++i) fprintf(o, "%.17g\n", b[i]);
    if (ferror(o)) die("cannot write solution file");
    if (fclose(o) != 0) die("cannot close solution file");

    klu_free_symbolic(&Sym, &Common);
    klu_free_numeric(&Num, &Common);
    free(Ap); free(Ai); free(Ax); free(b); free(res);
    return 0;
}
