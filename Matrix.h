#ifndef MATRIX
#define MATRIX
#include<bits/stdc++.h>
using namespace std;
typedef vector< vector<double> > DoubleVector2;

class Matrix{
public:
    // n rows, m columns
    pair<int, int> dimension;

    // vecs[i][j] denotes the element in i row j column
    DoubleVector2 vecs;

    // generate a vector with d empty vector<double>s.
    DoubleVector2 generateVector(int d) const{
        DoubleVector2 vec;
        for (int i = 0; i != d; i++){
            vector<double> v;
            vec.push_back(v);
        }
        return vec;
    }

    Matrix(){
        pair<int, int> p(0,0);
        this->dimension = p;
    }

    // the usual constructor
    Matrix(pair<int, int> dimension, DoubleVector2 vecs){
        this->dimension = dimension;
        this->vecs = vecs;
    }

    // constructor for an empty matrix with assigned dimension
    Matrix(pair<int, int> dimension){
        srand((unsigned)time(NULL));
        this->dimension = dimension;
        vecs = generateVector(dimension.first);
        for (int i = 0; i != dimension.first; i++)
            for (int j = 0; j != dimension.second; j++){
                double x = (rand() % 10000) / 10000.0;
                vecs[i].push_back(x);
            }
    }

    Matrix(vector<double> vec, int n){
        pair<int, int> p(n, vec.size());
        this->dimension = p;
        for (int i = 0; i != n; i++)
            vecs.push_back(vec);
    }

    // accessor to dimension
    pair<int, int> getDimension() const{
        return dimension;
    }

    // accessor to vecs
    DoubleVector2 getVectors() const{
        return vecs;
    }

    // transpose the matrix
    Matrix transpose() const{
        DoubleVector2 vec;
        for (int i = 0; i != vecs[0].size(); i++){
            vector<double> v;
            vec.push_back(v);
        }
        for (int i = 0; i != vecs.size(); i++){
            for (int j = 0; j != vecs[i].size(); j++)
                vec[j].push_back(vecs[i][j]);
        }
        pair<int, int> dimen(dimension.second, dimension.first);
        Matrix mat(dimen, vec);
        return mat;
    }

    // output the dimension of the matrix in the first row
    // then output the elements in the matrix
    void output() const{
        //cout << dimension.first << " " << dimension.second << endl;
        for (int i = 0; i != vecs.size(); i++){
            for (int j = 0; j != vecs[i].size(); j++)
                cout << vecs[i][j] << " ";
            cout << endl;
        }
    }

    // load the matrix elements with input number through cin
    void cin_input(){
        for (int i = 0; i != dimension.first; i++){
            for (int j = 0; j != dimension.second; j++){
                int x;
                cin>>x;
                vecs[i][j] = x;
            }
        }
    }

    // plus operation
    Matrix operator +(const Matrix &matrix) const{
        if(dimension != matrix.getDimension()){
            cout << "illegal plus operation" << endl;
            exit(0);
        }
        DoubleVector2 vec = matrix.getVectors();
        for (int i = 0; i != dimension.first; i++){
            for (int j = 0; j != dimension.second; j++)
                vec[i][j] += vecs[i][j];
        }
        Matrix m(dimension, vec);
        return m;
    }

    // minus operation
    Matrix operator -(const Matrix &matrix) const{
        if(dimension != matrix.getDimension()){
            cout << "illegal minus operation" << endl;
            exit(0);
        }
        DoubleVector2 vec = matrix.getVectors();
        for (int i = 0; i != dimension.first; i++){
            for (int j = 0; j != dimension.second; j++)
                vec[i][j] = vecs[i][j] - vec[i][j];
        }
        Matrix m(dimension, vec);
        return m;
    }

    // multiply operation
    Matrix operator *(const Matrix &matrix) const{
        if(dimension.second != matrix.getDimension().first){
            cout << "illegal multiply operation" << endl;
            cout << dimension.first << " " << dimension.second << endl;
            output();
            cout << endl;
            cout << matrix.dimension.first << " " << matrix.dimension.second << endl;
            matrix.output();
            exit(0);
        }
        Matrix m = matrix.transpose();
        DoubleVector2 vm = m.getVectors();
        pair<int, int> new_dimension(dimension.first,
                                     matrix.getDimension().second);
        DoubleVector2 v = generateVector(dimension.first);

        // i locates the row number of matrix1
        for (int i = 0; i != dimension.first; i++){
            // j locates the column number of matrix2
            for (int j = 0; j != matrix.getDimension().second; j++){
                double tmp = 0;
                // k locates the column#/row# of matrix1/matrix2
                for (int k = 0; k != dimension.second; k++)
                    tmp += vecs[i][k] * vm[j][k];
                v[i].push_back(tmp);
            }
        }
        Matrix mat(new_dimension, v);
        return mat;
    }

    Matrix operator *(const double &multi) const{
        Matrix m(dimension);
        for (int i = 0; i != dimension.first; i++){
            for (int j = 0; j != dimension.second; j++)
                m.vecs[i][j] = vecs[i][j] * multi;
        }
        return m;
    }

    Matrix point_multiply(const Matrix &matrix) const{
        if(dimension != matrix.getDimension()){
            cout << "illegal point multiply operation" << endl;
            exit(0);
        }

        Matrix mat(dimension);
        for (int i = 0; i != dimension.first; i++){
            for (int j = 0; j != dimension.second; j++)
                mat.vecs[i][j] = matrix.vecs[i][j] * vecs[i][j];
        }
        return mat;
    }
};
#endif // MATRIX
