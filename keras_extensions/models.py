import theano

from keras.models import Model, standardize_X
from keras.layers import containers
from keras import optimizers, objectives

class SingleLayerUnsupervised(Model, containers.Sequential):
    """
    Single layer unsupervised learning Model.
    """
    # add Layer, adapted from keras.layers.containers.Sequential
    def add(self, layer):
        if len(self.layers) > 0:
            warnings.warn('Cannot add more than one Layer to SingleLayerUnsupervised!')
            return
        super(SingleLayerUnsupervised, self).add(layer)

    # compile theano graph, adapted from keras.models.Sequential
    def compile(self, optimizer, loss, theano_mode=None):
        self.optimizer = optimizers.get(optimizer)

        self.loss = objectives.get(loss)
#        weighted_loss = weighted_objective(objectives.get(loss))

        # input of model
        self.X_train = self.get_input(train=True)
        self.X_test = self.get_input(train=False)

#        self.y_train = self.get_output(train=True)
#        self.y_test = self.get_output(train=False)

#        # target of model
#        self.y = T.zeros_like(self.y_train)

#        self.weights = T.ones_like(self.y_train)

#        train_loss = weighted_loss(self.y, self.y_train, self.weights)
#        test_loss = weighted_loss(self.y, self.y_test, self.weights)
        train_loss = self.loss(self.X_train)
        test_loss = self.loss(self.X_test)

        train_loss.name = 'train_loss'
        test_loss.name = 'test_loss'
#        self.y.name = 'y'

#        if class_mode == "categorical":
#            train_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_train, axis=-1)))
#            test_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_test, axis=-1)))

#        elif class_mode == "binary":
#            train_accuracy = T.mean(T.eq(self.y, T.round(self.y_train)))
#            test_accuracy = T.mean(T.eq(self.y, T.round(self.y_test)))
#        else:
#            raise Exception("Invalid class mode:" + str(class_mode))
        #train_accuracy = monitor(self.X_train)
        #test_accuracy = monitor(self.X_test)
#        self.class_mode = class_mode
        self.theano_mode = theano_mode

        for r in self.regularizers:
            train_loss = r(train_loss)
        updates = self.optimizer.get_updates(self.params, self.constraints, train_loss)

        if type(self.X_train) == list:
            train_ins = self.X_train# + [self.y, self.weights]
            test_ins = self.X_test# + [self.y, self.weights]
#            predict_ins = self.X_test
        else:
            train_ins = [self.X_train]#, self.y, self.weights]
            test_ins = [self.X_test]#, self.y, self.weights]
#            predict_ins = [self.X_test]

        self._train = theano.function(train_ins, train_loss,
            updates=updates, allow_input_downcast=True, mode=theano_mode)
        #self._train_with_acc = theano.function(train_ins, [train_loss, train_accuracy],
        #    updates=updates, allow_input_downcast=True, mode=theano_mode)
#        self._predict = theano.function(predict_ins, self.y_test,
#            allow_input_downcast=True, mode=theano_mode)
        self._test = theano.function(test_ins, test_loss,
            allow_input_downcast=True, mode=theano_mode)
        #self._test_with_acc = theano.function(test_ins, [test_loss, test_accuracy],
        #    allow_input_downcast=True, mode=theano_mode)

    # train model, adapted from keras.models.Sequential
    def fit(self, X, batch_size=128, nb_epoch=100, verbose=1, callbacks=[],
            validation_split=0., validation_data=None, shuffle=True, show_accuracy=False):
        X = standardize_X(X)
#        y = standardize_y(y)
#        sample_weight = standardize_weights(y, class_weight=class_weight, sample_weight=sample_weight)

        val_f = None
        val_ins = None
        if validation_data or validation_split:
            if show_accuracy:
                val_f = self._test_with_acc
            else:
                val_f = self._test
#        if validation_data:
#            try:
#                X_val, y_val = validation_data
#            except:
#                raise Exception("Invalid format for validation data; provide a tuple (X_val, y_val). \
#                    X_val may be a numpy array or a list of numpy arrays depending on your model input.")
#            X_val = standardize_X(X_val)
#            y_val = standardize_y(y_val)
#            val_ins = X_val + [y_val, np.ones(y_val.shape[:-1] + (1,))]

        if show_accuracy:
            f = self._train_with_acc
            out_labels = ['loss', 'acc']
        else:
            f = self._train
            out_labels = ['loss']

        ins = X# + [y, sample_weight]
        metrics = ['loss', 'acc', 'val_loss', 'val_acc']
        return self._fit(f, ins, out_labels=out_labels, batch_size=batch_size, nb_epoch=nb_epoch,
                         verbose=verbose, callbacks=callbacks,
                         validation_split=validation_split, val_f=val_f, val_ins=val_ins,
                         shuffle=shuffle, metrics=metrics)
