#include <stdlib.h>
#include <stdio.h>

#if 0
Usually, we have a dataset to learn from: something measured from the real
world, or an image, sound, whatever, and what we have to learn from it.

Here, since this is an example, we will generate our own data. That allows us
to make sure we learn correctly by comparing what we learn VS how we generate
the data.

In a real dataset, we first have to *normalize* input: center every variable
around 0, and make it go from -a to a, whatever the value is, is a good way to
do. Since we dont have any dataset, we will generate one already normalized,
where each variable is from -10 to 10. If we don't, gradients involving x
will take crazy high values, we'll have an "exploding gradient" problem, and
J and the gradient will quickly reach infinity or nan.
#endif

// useful function to generate a random number in a range
double randr(float from, float to) {
    double distance = to - from;
    return ((double)rand() / ((double)RAND_MAX + 1) * distance) + from;
}

// the hidden params
const int dataset_size = 100;
const float unknown_W[][3] = {
    { 20, -1, -3 },
    { 3, -2, 8 }
};

const float unknown_b[] = {
    2, 1
};

void generate_data(int nb_examples, double** y_ptr, double** x_ptr) {
    *y_ptr = malloc(nb_examples * sizeof (double) * 2);
    *x_ptr = malloc(nb_examples * sizeof (double) * 3);

    double* x = *x_ptr;
    double* y = *y_ptr;

    for (int i = 0; i < nb_examples; ++i) {
        // normalized values
        double a = randr(-10, 10);
        double b = randr(-10, 10);
        double c = randr(-10, 10);
        x[i * 3 + 0] = a;
        x[i * 3 + 1] = b;
        x[i * 3 + 2] = c;

        y[i * 2 + 0] =
            a * unknown_W[0][0]
            + b * unknown_W[0][1]
            + c * unknown_W[0][2]
            + unknown_b[0]
            + randr(-1, 1); // random error, noise
        y[i * 2 + 1] =
            a * unknown_W[1][0]
            + b * unknown_W[1][1]
            + c * unknown_W[1][2]
            + unknown_b[1]
            + randr(-1, 1); // random error, noise
    }
}

// model parameters
double W[][3] = {
    {0, 0, 0},
    {0, 0, 0}
};

double b[] = { 0, 0 };

void random_init_params() {
    W[0][0] = randr(-1, 1);
    W[0][1] = randr(-1, 1);
    W[0][2] = randr(-1, 1);

    W[1][0] = randr(-1, 1);
    W[1][1] = randr(-1, 1);
    W[1][2] = randr(-1, 1);

    b[0] = randr(-1, 1);
    b[1] = randr(-1, 1);
}

void print_params() {
    printf("W0,0 = %.2lf\tW0,1 = %.2lf\tW0,2 = %.2lf\n", W[0][0], W[0][1], W[0][2]);
    printf("W1,0 = %.2lf\tW1,1 = %.2lf\tW1,2 = %.2lf\n", W[1][0], W[1][1], W[1][2]);
    printf("b0 = %.2lf\tb1 = %.2lf\n", b[0], b[1]);
    printf("CORRECTS:\n");
    printf("W0,0 = %.2lf\tW0,1 = %.2lf\tW0,2 = %.2lf\n", unknown_W[0][0], unknown_W[0][1], unknown_W[0][2]);
    printf("W1,0 = %.2lf\tW1,1 = %.2lf\tW1,2 = %.2lf\n", unknown_W[1][0], unknown_W[1][1], unknown_W[1][2]);
    printf("b0 = %.2lf\tb1 = %.2lf\n", unknown_b[0], unknown_b[1]);
}

// Linear Regression
// h: output, predicted value
// x: input
void compute_hypothesis(double* h, double*x) {
        h[0] = x[0] * W[0][0] + x[1] * W[0][1] + x[2] * W[0][2] + b[0];
        h[1] = x[0] * W[1][0] + x[1] * W[1][1] + x[2] * W[1][2] + b[1];
}

// MSE
// h: predicted value
// y: real value
double mse(double* h, double* y) {
    double err0 = h[0] - y[0];
    double err1 = h[1] - y[1];
    return 0.5 * (err0 * err0 + err1 * err1);
}

// Update
// h: predicted value
// y: real value
// x: input
double update_params(double alpha, double* h, double* y, double* x) {
    W[0][0] -= alpha * ((h[0] - y[0]) * x[0]);
    W[0][1] -= alpha * ((h[0] - y[0]) * x[1]);
    W[0][2] -= alpha * ((h[0] - y[0]) * x[2]);

    W[1][0] -= alpha * ((h[1] - y[1]) * x[0]);
    W[1][1] -= alpha * ((h[1] - y[1]) * x[1]);
    W[1][2] -= alpha * ((h[1] - y[1]) * x[2]);

    b[0] -= alpha * (h[0] - y[0]);
    b[1] -= alpha * (h[1] - y[1]);
}

void SGD_one_pass(double* y, double* x) {
    double h[2]; // predicted values

    for (int i = 0; i < dataset_size; ++i) {
        double* this_x = &x[i * 3];
        double* this_y = &y[i * 2];
        compute_hypothesis(h, this_x);
        double J = mse(h, this_y);
        printf("loss: %lf\n", J);
        update_params(0.01, h, this_y, this_x);
    }
}

int main() {
    srand(42); // initialize random with a deterministic seed for testing

    double* x;
    double* y;
    generate_data(dataset_size, &y, &x);

    random_init_params();

    // learn 5 times from the dataset
    for (int i = 0; i < 5; ++i) {
        SGD_one_pass(y, x);
    }

    print_params();

    free(x);
    free(y);
    return 0;
}

