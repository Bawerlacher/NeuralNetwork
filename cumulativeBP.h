#ifndef CUMULATIVEBP
#define CUMULATIVEBP
#include<bits/stdc++.h>
#include "Matrix.h"
using namespace std;
typedef vector< vector<double> > DoubleVector2;
typedef pair<Matrix, Matrix> NN_Data;

Matrix activation(Matrix mat, int t);
Matrix d_activation(Matrix mat, int t);

class NeuralNetwork{
private:
    // n sets of test cases, m_i denotes how many values in the layer i
    // X: n * m_0 ; Y: n * m_(n-1)
    Matrix X, Y;
    // Z_i: m_i * n
    vector<Matrix> Z;
    // A_i: m_i * n
    vector<Matrix> A;
    // Delta_i: m_i * n
    vector<Matrix> Delta;
    // Omega_i: m_i * m_(i-1)
    vector<Matrix> Omega;
    // Theta_i: m_i * n
    vector<Matrix> Theta;

    vector<int> Size;
    int fType;
    double alpha;

public:
    NeuralNetwork(Matrix X, Matrix Y, vector<int> Size, double alpha, int fType = 0){
        this->X = X.transpose();
        this->Y = Y.transpose();
        this->Size = Size;
        this->fType = fType;
        this->alpha = alpha;
        ConstructNetwork();
    }

    void ConstructNetwork(){
        int n = X.getDimension().second;
        for(int i = 0; i != Size.size(); i++){
            int m = Size[i];
            pair<int, int> si(m, n);
            Matrix mat(si);
            Z.push_back(mat);
            A.push_back(mat);
            Delta.push_back(mat);
            Theta.push_back(mat);
            if (i == 0){
                A[0] = X;
                si.second = 0;
                Matrix matt(si);
                Omega.push_back(matt);
            }
            else{
                si.second = Size[i - 1];
                Matrix matt(si);
                Omega.push_back(matt);
            }
        }
    }

    void forward_propagation(){
        for (int i = 1; i != Size.size(); i++){
            Z[i] = Omega[i] * A[i - 1];
            A[i] = activation(Z[i] - Theta[i], fType);
        }
    }

    void backward_propagation(){
        for (int i = Size.size() - 1; i >= 1; i--){
            if (i == Size.size() - 1)
                Delta[i] = (A[i] - Y).point_multiply(d_activation(Z[i] - Theta[i], fType));
            else{
                Matrix ome = Omega[i + 1].transpose();
                Delta[i] = (ome * Delta[i + 1]).point_multiply(d_activation(Z[i] - Theta[i], fType));
            }
            Omega[i] = Omega[i] - Delta[i] * A[i - 1].transpose() * alpha;
            vector<double> theta = Theta[i].transpose().vecs[0];
            for (int j = 0; j < Delta[i].dimension.first; j++){
                double tmp = 0;
                for (int k = 0; k < Delta[i].dimension.second; k++){
                    tmp += Delta[i].vecs[j][k];
                }
                theta[j] += alpha * tmp;
            }
            Matrix Th(theta, Theta[i].dimension.second);
            Theta[i] = Th.transpose();
        }
    }

    void train(int iterations){
        for (int i = 0; i != iterations; i++){
            forward_propagation();
            backward_propagation();
        }
        cout << "training ends" << endl;
    }

    void show_network(){
        cout << "in layer0, the output is" << endl;
        A[0].output();
        for (int i = 1; i != Size.size(); i++){
            cout << "in layer" << i << ", the omega is" << endl;
            Omega[i].output();
            cout << "the input is" << endl;
            Z[i].output();
            cout << "the Theta is" << endl;
            Theta[i].output();
            cout << "the output is" << endl;
            A[i].output();
            cout << "the error is" << endl;
            Delta[i].output();
        }
    }

    void test(Matrix M){
        vector<Matrix> Th;
        for (int i = 0; i != Size.size(); i++){
            vector<double> theta = Theta[i].transpose().vecs[0];
            Matrix matt(theta, M.dimension.first);
            Theta[i] = matt.transpose();
        }
        A[0] = M.transpose();
        forward_propagation();
        show_network();
    }
};

NN_Data load_data(char* filename, int set_num, int x_num, int y_num){
    freopen(filename, "r", stdin);
    DoubleVector2 X, Y;
    double xx, yy;
    for (int i = 0; i != set_num ; i++){
        vector<double> x;
        vector<double> y;
        for (int j = 0; j != x_num; j++){
            cin >> xx;
            x.push_back(xx);
        }
        X.push_back(x);
        for (int j = 0; j != y_num; j++){
            cin >> yy;
            y.push_back(yy);
        }
        Y.push_back(y);
    }
    fclose(stdin);
    pair<int, int> p1(set_num, x_num);
    Matrix matX(p1, X);
    pair<int, int> p2(set_num, y_num);
    Matrix matY(p2, Y);
    NN_Data p(matX, matY);
    return p;
}

double inline m_sigmoid(double x){
    return 1 / (1 + exp(-x));
}

double inline d_sigmoid(double x){
    return m_sigmoid(x) * (1 - m_sigmoid(x));
}

double inline m_tanh(double x){
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

double inline d_tanh(double x){
    return 1 - m_tanh(x) * m_tanh(x);
}

Matrix activation(Matrix mat, int t){
    Matrix m(mat.dimension);
    for (int i = 0; i != mat.dimension.first; i++)
        for (int j = 0; j != mat.dimension.second; j++){
            if (t == 0)
                m.vecs[i][j] = m_sigmoid(mat.vecs[i][j]);
            else if(t == 1)
                m.vecs[i][j] = m_tanh(mat.vecs[i][j]);
        }
    return m;
}

Matrix d_activation(Matrix mat, int t){
    Matrix m(mat.dimension);
    for (int i = 0; i != mat.dimension.first; i++)
        for (int j = 0; j != mat.dimension.second; j++){
            if (t == 0)
                m.vecs[i][j] = d_sigmoid(mat.vecs[i][j]);
            else if(t == 1)
                m.vecs[i][j] = d_tanh(mat.vecs[i][j]);
        }
    return m;
}

#endif // CUMULATIVEBP
