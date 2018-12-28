#ifdef STANDARD_BP_NEURALNETWORK
#define STANDARD_BP_NEURALNETWORK

#include<bits/stdc++.h>
#define SIGMOID 0
#define TANH 1
#define L_INPUT 0
#define L_HIDDEN 1
#define L_OUTPUT 2
#define MAX_LAYER 20
#define INF 999999999
#define INTERVAL 1
#define NB 0.1
#define LBD 1000
#define INSPECT_SET 0
using namespace std;
typedef vector< vector<double> > DoubleVector2;
typedef pair<DoubleVector2, DoubleVector2> NN_Data;

/*this function denotes activation function, in which t denotes which function
 *to use. Right now, t = 0 uses sigmoid and t = 1 uses tanh. */
double activation(double x, int t);

/*this function calculate the derivative of an activation function*/
double d_activation(double x, int t);

/** This class simulates the structure of an artificial neuron. It stores the
 * weights that linking to the neuron, the input value, the threshold value,
 * the output value and the error.
 * it supports forward calculation, backward propagation, updating weights and refreshing.
 */
class Neuron{
public:
    double output, input, theta, error;

    // layer denotes whether it's in input layer, hidden layer or output layer.
    // fType denotes which activation function it would use.
    int layer, fType;
    vector<Neuron*> link;
    vector<double> weight;

    // constructor for neurons in hidden layer and output layer.
    Neuron(vector<Neuron*> link, int layer, int fType, int input = 0, int output = 0){
        this->link = link;
        this->layer = layer;
        this->fType = fType;
        this->input = input;
        this->output = output;
        theta = error = 0;
        for (int i = 0; i != link.size(); i++)
            weight.push_back(1);
    }

    // constructor for INPUT LAYER only(no link or weight)
    Neuron(int fType, int input = 0, int output = 0){
        this->layer = L_INPUT;
        this->fType = fType;
        this->input = input;
        this->output = output;
        theta = error = 0;
    }

    /* pick up the outputs of neurons in previous layer and multiply them with
     * their weight respectively. Then add them up together and generate output
     * through activation function. */
    void forward_process(){
        for (int i = 0; i != link.size(); i++)
            this->input += weight[i] * link[i]->output;
        if(layer != L_INPUT)
            this->output = activation(this->input - this->theta, fType);
    }

    // calculate the error for current neuron and propagate it to previous layer.
    void propagate_error(){
        error *= d_activation(input - theta, fType);
        for (int i = 0; i != link.size(); i++)
            link[i]->error += weight[i] * this->error;
    }

    // calculate errors for the last layer (output layer)
    void set_error(int y){
        if (layer != L_OUTPUT)
            return;
        error = y - output;
    }

    // use gradient descent
    void update(double alpha){
        for (int i = 0; i != link.size(); i++)
            weight[i] += this->error * link[i]->output * alpha;

        theta -= alpha * error;
    }

    // reinitialize the input, output and error.
    void refresh(){
        input = output = error = 0;
    }
};


/**
 * this class simulates the structure of a neural network, which constitutes
 * a bunch of neurons that linking each other by layer.
 * it stores the training data with X as input and with Y as output.
 * it support standard easy BP algorithm to optimize parameters.
 */
class NeuralNetwork{
private:
    // Size denote how many layers in the NN and how many neurons in each layer.
    vector<int> Size;

    // this store the addresses of each neurons by layer.
    vector<Neuron*> neurons[MAX_LAYER];

    vector< vector<double> > X;
    vector< vector<double> > Y;

    // fType denotes which activation function to be used.
    int fType;

    // alpha is the learning rate.
    double alpha;

public:

    NeuralNetwork(vector<int> Size, vector< vector<double> > input,
                   vector< vector<double> > output, double alpha, int fType = 0){
        this->Size = Size;
        this->X = input;
        this->Y = output;
        this->alpha = alpha;
        this->fType = fType;
        construct_network();
    }

    // for testing
    NeuralNetwork(vector<int> Size, double alpha, int fType = 0){
        this->Size = Size;
        this->alpha = alpha;
        this->fType = fType;
        construct_network();
    }

    /* this method constructs the network by setting neurons from input layer to
     * output layer. The link of each neuron is the set of pointers of neurons in
     * the previous layer. */
    void construct_network(){
        for (int i = 0; i < Size.size(); i++){
            for (int j = 0; j < Size[i]; j++){
                Neuron* p;

                // i == 0 means it's in the input layer
                if (i == 0)
                    p = new Neuron(fType);
                else if (i == Size.size() - 1)
                    p = new Neuron(neurons[i - 1], L_OUTPUT, fType);
                else
                    p = new Neuron(neurons[i - 1], L_HIDDEN, fType);

                neurons[i].push_back(p);
            }
        }
    }

    /* used for refreshing each neurons in the network */
    void refresh_network(){
        for (int i = 0; i != Size.size(); i++){
            for (int j = 0; j != neurons[i].size(); j++){
                neurons[i][j]->refresh();
            }
        }
    }

    /* do forward propagation here */
    void forward_propagation(vector<double> in){
        // initialize the input layer
        for (int i = 0; i != in.size(); i++)
            neurons[L_INPUT][i]->output = in[i];

        // forward propagation
        for (int i = 1; i != Size.size(); i++)
            for (int j = 0; j != neurons[i].size(); j++)
                neurons[i][j]->forward_process();
    }

    /* do backward propagation and update parameters */
    void backward_propagation(vector<double> out){
        // from the output layer to input layer(which is excluded)
        for (int i = Size.size() - 1; i > 0; i--){
            for (int j = 0; j != Size[i]; j++){
                if (i == Size.size() - 1){
                    neurons[i][j]->set_error(out[j]);
                }

                // calculate error and propagate it here
                neurons[i][j]->propagate_error();

                neurons[i][j]->update(alpha);
            }
        }
    }

    // calculate the output of cost function
    double cal_cost(vector<double> out){
        double cost = 0;
        for (int i = 0; i != out.size(); i++)
            cost += pow(out[i] - neurons[Size.size() - 1][i]->output, 2);
        return cost / 2;
    }

    // train parameters here
    void train(){
        // iterations for each example in the training set.
        for (int i = 0; i != X.size(); i++){
            double last_cost, cur_cost;
            cur_cost = INF;
            int lambda = 0;
            do{
                lambda++;
                last_cost = cur_cost;
                refresh_network();
                forward_propagation(X[i]);
                cur_cost = cal_cost(Y[i]);
                backward_propagation(Y[i]);
                if (cur_cost > last_cost + INTERVAL){
                    cout<<"Oh no! Cost is increasing!"<<i<<" "<<cur_cost<<" "<<last_cost<<endl;
                    return;
                }
            }while(lambda < LBD); //LBD: the limit of iteration
        }

        cout<<"training ends"<<endl;
    }

    // show all the data in the neural network (for the use of testing)
    void show_network(){
        for (int i = 0; i != Size.size(); i++){
            for (int j = 0; j != Size[i]; j++){
                cout<<"for neurons ["<<i<<", "<<j<<"]"<<endl;
                cout<<"input: "<<neurons[i][j]->input
                    <<" output: "<<neurons[i][j]->output
                    <<" theta: "<<neurons[i][j]->theta
                    <<" error: "<<neurons[i][j]->error<<endl;
                vector<double> w = neurons[i][j]->weight;
                if(i == 0)continue;
                cout<<"Weights: ";
                for (int k = 0; k != w.size(); k++)
                    cout<<"to"<<k<<" "<<w[k]<<", ";
                cout<<endl;
            }
        }
        cout<<endl;
    }

    // test the neural network with testing data
    void test(vector<double> in){
        refresh_network();
        forward_propagation(in);
        for (int i = 0; i != Size[Size.size() - 1]; i++)
            cout << neurons[Size.size() - 1][i]->output << " ";
        cout<<endl;
    }

};

/** a function for loading data from a file, in which each test case takes
 *  a row with x_num input values and y_num output values. */
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
    NN_Data p(X, Y);
    return p;
}

///** activation functions **///

double inline sigmoid(double x){
    return 1 / (1 + exp(-x));
}

double inline d_sigmoid(double x){
    return sigmoid(x) * (1 - sigmoid(x));
}

double inline tanh(double x){
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

double inline d_tanh(double x){
    return 1 - tanh(x) * tanh(x);
}

double activation(double x, int t){
    if (t == 0)
        return sigmoid(x);
    else if (t == 1)
        return tanh(x);
    return INF;
}

double d_activation(double x, int t){
    if (t == 0)
        return d_sigmoid(x);
    else if (t == 1)
        return d_tanh(x);
    return INF;
}

#endif // STANDARD_BP_NEURALNETWORK
