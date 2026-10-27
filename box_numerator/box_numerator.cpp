#include "./mpreal/mpreal.h"
#include "./box_numerator.hpp"
#include <iostream>

using mpfr::mpreal;

BoxNumeratorEvaluator* create_evaluator_box4() {
    const mpreal ALPHA[4] = {
        -sqrt(mpreal("0.6")),
        mpreal("0.5")/sqrt(mpreal("0.6")),
        mpreal("0.0"),
        (mpreal("0.1")*sqrt(mpreal("11")))/sqrt(mpreal("0.6"))
    };
    const mpreal BETA[4] = {
        sqrt(mpreal("0.6")),
        -sqrt(mpreal("0.6")),
        0.0,
        0.0
    };
    const mpreal GAMMA[4] = {
        0.0,
        -mpreal("0.5")/sqrt(mpreal("0.6")),
        0.0,
        (-mpreal("0.1")*sqrt(mpreal("11")))/sqrt(mpreal("0.6"))
    };
    const mpreal M1 = 0;
    const mpreal M2 = 0;
    const mpreal M3 = 0;
    const mpreal M4 = 0;
    const mpreal S12 = -mpreal("0.2");
    const mpreal S13 = mpreal("0.6");
    const mpreal M32 = -mpreal("0.6");
    const mpreal PI_EXTENDED = mpfr::const_pi();

    return new BoxNumeratorEvaluator(ALPHA, BETA, GAMMA, M1, M2, M3, M4, M32, S12, S13, PI_EXTENDED);
}

BoxNumeratorEvaluator* create_evaluator_box5() {
    const mpreal ALPHA[4] = {
        -sqrt(mpreal("0.9")),
        sqrt(mpreal("0.73")/mpreal("0.9")),
        mpreal("0.0"),
        (mpreal("0.1")*sqrt(mpreal("8")))/sqrt(mpreal("0.9"))
    };
    const mpreal BETA[4] = {
        sqrt(mpreal("0.9")),
        -sqrt(mpreal("0.9")),
        mpreal("0.0"),
        mpreal("0.0")
    };
    const mpreal GAMMA[4] = {
        mpreal("0.0"),
        -sqrt(mpreal("0.73")/mpreal("0.9")),
        mpreal("0.0"),
        -(mpreal("0.1")*sqrt(mpreal("8")))/sqrt(mpreal("0.9"))
    };
    const mpreal M1 = 0;
    const mpreal M2 = 0;
    const mpreal M3 = 0;
    const mpreal M4 = 0;
    const mpreal S12 = 2*sqrt(mpreal("0.73"))-mpreal("1.8");
    const mpreal S13 = mpreal("0.9");
    const mpreal M32 = -mpreal("0.9");
    const mpreal PI_EXTENDED = mpfr::const_pi();

    return new BoxNumeratorEvaluator(ALPHA, BETA, GAMMA, M1, M2, M3, M4, M32, S12, S13, PI_EXTENDED);
}

BoxNumeratorEvaluator* create_evaluator_box6() {
    const mpreal ALPHA[4] = {
        -mpreal("1.0"),
        mpreal("0.4"),
        mpreal("0.0"),
        sqrt(mpreal("0.84"))
    };
    const mpreal BETA[4] = {
        mpreal("1.0"),
        -sqrt(mpreal("1.0")),
        0.0,
        0.0
    };
    const mpreal GAMMA[4] = {
        mpreal("0.0"),
        -mpreal("0.4"),
        mpreal("0.0"),
        -sqrt(mpreal("0.84"))
    };
    const mpreal M1 = 0;
    const mpreal M2 = 0;
    const mpreal M3 = 0;
    const mpreal M4 = 0;
    const mpreal S12 = -mpreal("1.2");
    const mpreal S13 = mpreal("1");
    const mpreal M32 = -mpreal("1");
    const mpreal PI_EXTENDED = mpfr::const_pi();

    return new BoxNumeratorEvaluator(ALPHA, BETA, GAMMA, M1, M2, M3, M4, M32, S12, S13, PI_EXTENDED);
}

extern "C" void integrate(
    int problem_index,
    int point_count,
    double *coordinates, 
    double *weights,
    double *out
) {
    const int digits = 100;
    mpreal::set_default_prec(mpfr::digits2bits(digits));

    mpreal accumulator = 0.0;

    BoxNumeratorEvaluator* compute_engine;
    if(problem_index == 4) {
        compute_engine = create_evaluator_box4();
    } else if (problem_index == 5) {
        compute_engine = create_evaluator_box5();
    } else if (problem_index == 6) {
        compute_engine = create_evaluator_box6();
    }


    for(int i = 0; i < point_count; i++) {
        const mpreal point[3] = {
            coordinates[i*3 + 0],
            coordinates[i*3 + 1],
            coordinates[i*3 + 2]
        };
        const mpreal weight = weights[i];
        compute_engine->eval(accumulator, point, weight);
    }

    mpreal corrected_accumulator = accumulator * mpreal("16") * mpfr::sqr(mpfr::const_pi());

    for(int i = 0; i < 10; i++) {
        out[i] = corrected_accumulator.toDouble(MPFR_RNDN);
        corrected_accumulator -= out[i];
    }

    delete compute_engine;
}