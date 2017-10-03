import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as mpl
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

'''
http://www.jessicayung.com/explaining-tensorflow-code-for-a-multilayer-perceptron/
'''

'''
This method reads the CSV file using as a parameter the filepath. It takes 2 columns: date and prices.
It automatically standartises the data with the method scale().
E.i. the length 1200 items
the Data reshaped in the following format:
[[0.432]
 [0.343]
 ...
 [0.876]]
'''
def read_file(path_to_file):

    raw_data = np.loadtxt(path_to_file, delimiter=",", skiprows=1, usecols=(1,5))
    raw_data = scale(raw_data)

    #y is the dependent variable
    samples = raw_data[:, 1].reshape(-1, 1)

    return samples

'''
The following method takes the samples from read_file() method and
converts the values according to input and output layers.
e.i. [4,2,2,1] where 4 neurons in input layer, 1 neuron in output layer
'''
def set_up_data(rawData, layers):

    number_samples = len(rawData) #number of samples
    input_neurons = layers[0]
    output_neurons = layers[-1]

    #train sise should be divisible by the number of neurons
    #1200 / 4 * 4 = 1200 or 1200/11*11 = 1199

    train_size = int(number_samples/input_neurons)*input_neurons
    #print("train size is", train_size)
    train_data = []
    resized_data = np.array(rawData)[:train_size+1] #data is resized in accordance with the  train_size
    '''
    The following loop is used in order to populate the data. Let's say we want to use 4 past days as input
    in order to predict the 5th day value. The array of prices looks like [0,1,2,3,4,5,6,7,8,9,10,11,12].
    The required format for MLP graph looks like:
    input                   output
    [[0,1,2,3]              [[4]
     [1,2,3,4]               [5]
     [2,3,4,5]               [6]
     ...                     ...
     [8,9,10,11]]            [12]]

     Length of the train data  = train_data - (number of input neurons)
     In the mentioned example the size of train_data = 1196
     In the mentioned example the size of target_data = 1196
    '''

    for i in range(len(resized_data) - input_neurons):
        for j in range(input_neurons):
            train_data.append(resized_data[i+j])

    train_data = np.reshape(train_data, [-1, input_neurons])
    #print('length of train data is ', len(train_data))

    target_data = np.array(rawData)[input_neurons: train_size+1]
    target_data = target_data.reshape(-1, output_neurons)
    num = len(target_data)
    #print('target data is', len(target_data))

    return train_data, target_data


def create_biases(list_layers):
    '''
    Additional method to create a bias vector for an MLP.
    :param list_layers: list that specifies the number of neurons in each layer
    :return: vector of biases
    '''
    biases = []
    n = len(list_layers)
    for i in range(n - 1):
        s_dev = np.sqrt(1.0 / list_layers[i])
        b = tf.Variable(tf.random_normal([list_layers[i + 1]], stddev=s_dev))
        biases.append(b)
    return biases

def create_weights(list_layers):
    '''
    Additional method to create a weight matrix for an MLP.
    @:param s_dev - standard deviation
    @:param list_layers: a list that specifies the number of neurons in each layer
    @:param return: matrix of weights
    '''
    weights = []
    n = len(list_layers)
    for i in range(n - 1):
        s_dev = np.sqrt(1.0 / list_layers[i])
        w = tf.Variable(tf.random_normal([list_layers[i], list_layers[i + 1]], stddev=s_dev))
        weights.append(w)
    return weights


def select_act_func(name):
    '''
    Additional method to select the activation function
    :param name - the custom selection of the activation function
    :return the tf activation function implementation
    '''
    if name == 'sigmoid':
        return tf.sigmoid
    elif name == 'tanh':
        return tf.tanh
    elif name == 'elu':
        return tf.nn.elu
    elif name == 'softplus':
        return tf.nn.softplus
    elif name == 'softsign':
        return tf.nn.softsign
    elif name == 'relu':
        return tf.nn.relu
    elif name == 'relu6':
        return tf.nn.relu6
    return None

def select_loss_func(name):
    '''
    additional method to select the loss function
    :param name: name of the loss function
    :return: the tf realization of the function
    '''
    if name == 'l2':
        return lambda YH, Y : tf.squared_difference(Y, YH)
    elif name == 'l1':
        return lambda YH, Y : tf.losses.absolute_difference(Y, YH)
    elif name == 'smce':
        return lambda YH, Y : tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = YH)
    elif name == 'sgce':
        return lambda YH, Y : tf.nn.sigmoid_cross_entropy_with_logits(labels = Y, logits = YH)
    elif name == 'cos':
        return lambda YH, Y : tf.losses.cosine_distance(Y, YH)
    elif name == 'log':
        return lambda YH, Y : tf.losses.log_loss(Y, YH)
    elif name == 'hinge':
        return lambda YH, Y : tf.losses.hinge_loss(Y, YH)
    return None


def select_optimizer(name, learn_rate):
    '''
        additional function to choose the optimizer
        :param name: name of the optimizer
        :param lr: learning rate of the network
        :return: tf optimizer function
        '''
    if name == 'adam':
        return tf.train.AdamOptimizer(learning_rate = learn_rate)
    elif name == 'gradient_descent':
        return tf.train.GradientDescentOptimizer(learning_rate = learn_rate)
    elif name == 'adagrad':
        return tf.train.AdagradOptimizer(learning_rate = learn_rate)
    elif name == 'ftrl':
        return tf.train.FtrlOptimizer(learning_rate = learn_rate)
    return None


def feed_forward(inputs, weights, biases, act_func):
    '''
    This method creates the MLP graph
    :param inputs: matrix of input neurons
    :param weights: weight matrix
    :param biases: bias vector
    :param act_func: selected activation function
    :return: the output (result of the feed forwarded MLP)
    '''
    n = len(weights)
    for i in range(n-1):
        inputs = act_func(tf.matmul(inputs, weights[i])+biases[i])
    return tf.matmul(inputs, weights[n - 1]) + biases[n - 1]

'''
This class initialises MLP, trains the net and predicts the values.
'''
class multilayer_perceptron:

    def __init__(self, list_layers, act_func, batch_size, learn_rate, loss_func, max_iter,
                 optimizer, path):
        '''
        MLP object initializer
        :param list_layers: array that contains the number of neurons in each layer, where the zero element is for input layer,
        the last element is for output layer, numbers in between specifies the number of neurons in hidden layers, e.i. [2,4,4,4,1]
        :param act_func: selected activation function
        :param batch_size: number of samples that are used as one batch in the training process
        :param learn_rate: - learning rate
        :param loss_func: - selected loss function. They measure the difference between the model outputs
        and the target (truth) values
        :param max_iter: specified maximum number of iterations
        :param optimizer: - selected optimizer
        :param path: - path to file
        '''
        self.path_file_temp = path
        self.layers = list_layers
        self.input_neurons = tf.placeholder("float", [None, self.layers[0]])
        self.output_neurons = tf.placeholder("float", [None, self.layers[-1]])
        self.weights = create_weights(self.layers)
        self.biases = create_biases(self.layers)
        self.act_func = select_act_func(act_func)

        self.data_train, self.data_target = set_up_data(read_file(self.path_file_temp), self.layers)

        self.epochs = max_iter
        self.batch_size = batch_size
        self.LF = select_loss_func(loss_func)
        self.y_hat = feed_forward(self.input_neurons, self.weights, self.biases, self.act_func)
        self.loss = tf.reduce_mean(tf.squared_difference(self.y_hat, self.output_neurons))
        self.optimizer = select_optimizer(optimizer, learn_rate).minimize(self.loss)

        self.RunSession()

    def RunSession(self):
        '''
        Start the TensorFlow session
        '''
        self.saver = tf.train.Saver()               #For saving a model for later restoration
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True      #Tell TensorFlow to use GPU memory as needed
        self.sess = tf.Session(config = config)     #instead of allocating all up-front
        init = tf.global_variables_initializer()
        self.sess.run(init)


    def train(self, data_train, data_target):
        '''
        train the MLP
        :param data_train: training data
        :param data_target: target data
        :return:
        '''
        m = len(data_train)
        err = 0
        for i in range(self.epochs):
            if self.batch_size is None:
                err, _ = self.sess.run([self.loss, self.optimizer], feed_dict={self.input_neurons:data_train, self.output_neurons:data_target})
            else:
                for j in range (0, m, self.batch_size):
                    bi = np.random.randint(m, size=self.batch_size)

                    l,_=self.sess.run([self.loss, self.optimizer], feed_dict={self.input_neurons:data_train[bi], self.output_neurons:data_target[bi]})
                    err+=l
                err /=len(range(0, m, self.batch_size))
        print('the error is', err)
        return err


    def predict(self, inputs):
        '''
        predict the next day' value
        :param inputs: number of neurons as a historical bunch based on which there will a prediction for the next day
        :return: predicted result
        '''

        if self.sess is None:
            raise Exception("Error: MLP has not yet been fitted.")
        return self.sess.run(self.y_hat, feed_dict = {self.input_neurons: inputs})

    def predict_for7days(self, data):
        '''
        This method computes predicted values for next 7 days
        !!! important - change 2 to the number of neurons
        :param data: input historical data
        :return: predicted values for 7 days
        '''
        #take the last 2 historical prices
        l = len(data)
        data = data[l-5:l]
        #create new array where we append the chosen number of las historical values
        arr = []
        for i in range(len(data)):
            arr.append(data[i])
        #predict values for 7 days ahead by looping
        for i in range(7):
            temp = np.asarray(arr[len(arr)-5:len(arr)])
            temp = np.reshape(temp, [-1, 5]) #reshape to the necessary format [[],[]]
            res = self.sess.run(self.y_hat, feed_dict = {self.input_neurons: temp})
            arr.append(res)
        return arr[5:]

    def get_dates(self):
        '''
        get date in scaled format from the CSV file
        !!! IMPORTANT in skiprows it's necessary to add the number of neurons
        the number of dates train_num - input_neurons.
        the date is provided from the n_th element, where n - number of neurons. if input_neurons = 2, then
        there will be no predicted results for the first two items, for the date array is [input_neurons: -1]
        :return:
        '''
        file_data = np.loadtxt(self.path_file_temp, delimiter=",", skiprows=(1), usecols=(1))
        train_size = int(len(file_data)/5)*5
        file_data = np.array(file_data)[5: train_size+1]
        file_data =scale(file_data)
        dates = file_data.reshape(-1, 1)

        #print(len(dates), 'amount of dates')
        return dates

    '''
    Plots the historical and predicted values on the graph
    '''
    def plot_results(self):

        dates = self.get_dates()
        yHat= self.predict(self.data_train)
        #print('length o predicted values', len(yHat))

        #Plot the results
        mpl.title(self.act_func)
        mpl.plot(dates, self.data_target, c='#b0403f')
        mpl.plot(dates, yHat, c='#5aa9ab')
        mpl.show()

    '''
    Method returns the accuracy between target values and predicted ones
    http://scikit-learn.org/stable/modules/model_evaluation.html  
    '''
    def calc_accuracy(self, target, predicted):
        mse = mean_squared_error(target, predicted)
        accuracy = (1 - mse)
        return accuracy

    def retrieve_result(self):
        #read date and values from the CSV file
        dates = np.loadtxt(self.path_file_temp, delimiter=",", skiprows=1+5, usecols=(1))
        hist_values = np.loadtxt(self.path_file_temp, delimiter=",", skiprows=1+5, usecols=(5))

        #!!! important - change 2 to the number of neurons
        train_size = int(len(hist_values)/5)*5
        hist_values = np.array(hist_values)[:train_size+1]
        hist_values = hist_values.reshape(len(hist_values), 1)

        #use StandardScaler to scale the data
        scaler = StandardScaler()
        scaler =scaler.fit(hist_values)
        #print('Mean: %f, StandardDeviation: %f' % (scaler.mean_, np.sqrt(scaler.var_)))

        #standartise the original target data
        standartised = scaler.transform(hist_values)
        #get the predicted values
        predicted = self.predict(self.data_train)

        #inverse transform the standartised predicted values to real values
        inversed_predicted = scaler.inverse_transform(predicted)
        predicted_7days = self.predict_for7days(self.data_target)
        predicted_7days = scaler.inverse_transform(predicted_7days)
        accuracy = self.calc_accuracy(self.data_target,predicted)*100
        print(accuracy, 'accuracy is')
        # for i in range (100):
        #     print(hist_values[i], '\t', inversed_predicted[i])
        return hist_values, inversed_predicted, inversed_predicted[-1], predicted_7days, accuracy


    def GetSession(self):
        if multilayer_perceptron.sess is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True      #Only use GPU memory as needed
            multilayer_perceptron.sess = tf.Session(config = config)      #instead of allocating all up-front
        return multilayer_perceptron.sess

    def save_model(self, p):
        '''
        Save a model to the file with the path "path/"
        :param p: path to save to
        '''
        self.saver.save(self.getSession(), p)






def launch_sess(path_to_file, list_layers, act_func, batch_size, learn_rate, loss_func, max_iter, optimizer):
    raw_data = read_file(path_to_file)
    mlp = multilayer_perceptron(list_layers=list_layers, act_func=act_func, batch_size=batch_size, learn_rate=learn_rate, loss_func=loss_func, max_iter=max_iter, optimizer=optimizer, path=path_to_file)
    data_train, data_target = set_up_data(raw_data, mlp.layers)
    #train the mlp graph
    mlp.train(mlp.data_train, mlp.data_target)
    mlp.plot_results()
    mlp.retrieve_result()
    







launch_sess('/Users/valeria/Desktop/Study/Project/yahoo.csv', list_layers=[5,10, 10,10,10, 10, 10, 1], act_func="tanh", batch_size=100, learn_rate=0.001, loss_func='l2', max_iter=500, optimizer='adam')