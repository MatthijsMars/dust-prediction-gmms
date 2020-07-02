from sklearn import mixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import multiprocessing
import joblib
import time
from functools import partial
from scipy.optimize import minimize
from tqdm import tqdm
import os
import pickle
from astropy.io import fits

class gmm(mixture.GaussianMixture):
    def __init__(self, whiten=False, save_name="", **kwargs):
        super().__init__(**kwargs)
        
        self.whiten = whiten
        if save_name != "":
            self.save_name = os.path.join("./models/", save_name)
            self.save = True
        else:
            self.save = False
        if self.whiten:
            self.scaler = StandardScaler()
            # self.scaler = MinMaxScaler()

        else:
            self.scaler = None

    def fit(self, X, y=None):
 
        if self.save:
            try: 
                print("try loading model")
                loaded_model = self.load_model(self.save_name)
                print("loaded model succesfully")
                return loaded_model
            except:
                print("model not found, training new model")
        
        print("training model")
        if self.whiten:
            X = self.scaler.fit_transform(X)
        super().fit(X, y) #TODO y scaling?
        if self.save:
            print(f"saving model to {self.save_name}")
            self.save_model(self.save_name)

        return self 

    def predict(self, X):
        if self.whiten:
            X = self.scaler.transform(X)
        super().predict(X)
    

    def save_model(self, filename):
        pickle.dump(self, open(filename, "wb"))

    def load_model(self, filename):
        loaded_model = pickle.load(open(filename, "rb"))
        print(loaded_model.__dict__)
        # self = loaded_model
        return loaded_model
    
    # @classmethod
    def class_parameters(cls, x_train, m, s, w, inv_s, det_s, new_s, B):
        cls.X = x_train 
        cls.means_ = m       
        cls.covariances_ = s       
        cls.weights_ = w       
        cls.inv_s = inv_s   
        cls.det_s = det_s   
        cls.new_s = new_s   
        cls.B = B      
        # print("X: ", cls.X)
        # cls.output = output
  
        
    def pdf_optimum(self, X, shape, limit=None, field=None, fname="", n_jobs=48):
        """find the optimum of the pdf for the last map, given the first maps"""
        X = self.scaler.transform(X)

        m, s, w = self.means_, self.covariances_, self.weights_
        self.inv_s = np.linalg.inv(s[:,:-1,:-1])
        self.B = (s[:,-1:,:-1] @ self.inv_s)
        self.new_s = s[:,-1:,-1:] - (self.B @ s[:,:-1,-1:] )
        self.det_s = np.linalg.det(s[:,-1:,-1:])

        result = np.zeros_like(X[:,-1])

        if field is not None:
            indices = np.arange(len(X),dtype=int)[field]
        else: 
            indices = np.arange(len(X), dtype=int)

        if limit is not None:
            indices = indices[:limit]

        st = time.time()
        print("starting..")
        chunks = 100
        # f = open("./logs/log " + fname[:-4] +".txt", 'w')
        # f.write("starting.. \n")
        # f.close()

        # result = np.zeros_like(X[:,-1])
        pool = multiprocessing.Pool(n_jobs)

        ds_m, ds_s = self.scaler.mean_[-1], self.scaler.scale_[-1]
        # ds_m, ds_s = self.scaler.data_min_[-1], 1/self.scaler.scale_[-1] #descale parameters
        for j in range(chunks):
            loop_indices = indices[int(j*len(X)/chunks):int((j+1)*len(X)/chunks)]
            st1 = time.time()
            self.chunk = X[loop_indices]

            for idx, x in pool.map(partial(_procedure, x_train=self.chunk, m=self.means_, s=self.covariances_, w=self.weights_, inv_s=self.inv_s, det_s=self.det_s, new_s=self.new_s, B=self.B),  np.arange(len(self.chunk)).astype(int)):
                result[loop_indices[0] +idx] = x*ds_s + ds_m
            if fname !='' and j % (chunks//10) == 0:
                fits.writeto("./data/processed/" + fname, result.reshape(shape), overwrite=True)
            # f = open("./logs/log " + fname[:-4] +".txt", 'a')
            print("finished predicting chunk {}, it took {:2f} min \n".format(j, (time.time()-st1)/60))
            # f.close()
        print("finished predicting, it took {:2f} min".format( (time.time()-st)/60))
        print(result)
        # print( np.sqrt(np.mean(np.square(self.scaler.inverse_transform(result) - X[:,-1])[indices])))
        if fname !='':
            fits.writeto("./data/processed/" + fname, result.reshape(shape), overwrite=True)
        return result

#    # def _procedure(self, i ):
    #     m, s, w = self.model.means_, self.model.covariances_, self.model.weights_
    #     # x_train, inv_s, det_s, new_s = self.x_train, self.inv_s, self.det_s, self.new_s
    #     y = self.chunk[i]
    #     dy = (y[:-1]-m[:,:-1]).reshape(len(m), len(y)-1, 1)

    #     new_m = m[:,-1:].reshape(len(m),1,1) + self.B @ dy

    #     ex1 =  -.5 * dy.reshape(len(m), 1, len(y)-1) @ (self.inv_s @ dy.reshape(len(m), len(y)-1, 1))
    #     K = 1/ np.sqrt( np.power(2*np.pi, len(m)) * det_s)

    #     new_w  = w.reshape(-1,1,1) * K.reshape(-1,1,1) * np.exp(ex1).reshape(-1,1,1)
    #     new_w /= np.sum(new_w)

    #     f = lambda y : -1*np.sum( gmm_pdf(y, new_m[:,0], self.new_s[:,0,0], new_w[:,0] ))
    #     # return minimize(f, np.median(y[:-1]), method='Powell').x

    #     return i, minimize(f, np.median(y[:-1]), method='Powell').x


def _procedure(i, x_train, m, s, w, inv_s, det_s, new_s, B):
    # m, s, w = self.model.means_, self.model.covariances_, self.model.weights_
    # x_train, inv_s, det_s, new_s = self.x_train, self.inv_s, self.det_s, self.new_s
    y = x_train[i]
    dy = (y[:-1]-m[:,:-1]).reshape(len(m), len(y)-1, 1)

    new_m = m[:,-1:].reshape(len(m),1,1) + B @ dy

    ex1 =  -.5 * dy.reshape(len(m), 1, len(y)-1) @ (inv_s @ dy.reshape(len(m), len(y)-1, 1))
    K = 1 / np.sqrt ( np.power(2*np.pi, len(m)) * det_s )

    new_w  = w.reshape(-1,1,1) * K.reshape(-1,1,1) * np.exp(ex1).reshape(-1,1,1)
    new_w /= np.sum(new_w)

    f = lambda y : -1*np.sum( gmm_pdf(y, new_m[:,0], new_s[:,0,0], new_w[:,0] ))
    # return minimize(f, np.median(y[:-1]), method='Powell').x
    return i, minimize(f, np.median(y[:-1]), method='Powell').x


#def 
    #     if self.whiten:
    #         X = self.scaler.transform(X)

    #     if limit is None:
    #         limit = len(X)
    #     # X = X
    #     inv_s = np.linalg.inv(self.covariances_[:,:-1,:-1])
    #     B = (self.covariances_[:,-1:,:-1] @ inv_s)
    #     new_s = self.covariances_[:,-1:,-1:] - (B @ self.covariances_[:,:-1,-1:] )
    #     det_s = np.linalg.det(self.covariances_[:,-1:,-1:])



    #     result = np.zeros_like(X[:,-1])

    #     indices = np.arange(len(X), dtype=int)

    #     st = time.time()
    #     print("starting..")
    #     chunks = 100


    #     if self.scaler is not None:
    #         ds_m, ds_s = self.scaler.mean_[-1], self.scaler.scale_[-1]
    #     else: 
    #         ds_m, ds_s = 0, 1


    #     #trying some different methods of parallelising
        
        
    #     # np.save("./" + fname, result)
    #     # print(fname)
    #     # joblib.dump(result, fname)
    #     # result = joblib.load(fname, mmap_mode='r+')

    #     # joblib.Parallel(n_jobs=48, backend="multiprocessing")(joblib.delayed(partial(_procedure, x_train=X, m=self.means_, s=self.covariances_, w=self.weights_, inv_s=self.inv_s, det_s=self.det_s, new_s=self.new_s, B=self.B, output=result))(i) for i in tqdm(indices))
            
    #     # joblib.dump(X, "X_memmap")
    #     # x_train = joblib.load("X_memmap", mmap_mode='r')

    #     # joblib.externals.loky.set_loky_pickler('pickle')

    #     # pool.map(self._procedure,  np.arange(len(indices)).astype(int))
    #     self.class_parameters(X, self.means_, self.covariances_, self.weights_, inv_s, det_s, new_s, B)

    #     if field is not None:
    #         indices = np.arange(len(self.x_train),dtype=int)[field]
    #     else: 
    #         indices = np.arange(len(self.x_train), dtype=int)

    #     if limit is not None:
    #         indices = indices[:limit]

    #     ds_m, ds_s = self.Xscaler.mean_[-1], self.Xscaler.scale_[-1]
    #     for j in range(chunks):
    #         loop_indices = indices[int(j*len(self.x_train)/chunks):int((j+1)*len(self.x_train)/chunks)]
    #         st1 = time.time()
    #         self.chunk = self.x_train[loop_indices]

    #         for idx, x in pool.map(partial(_procedure, x_train=self.chunk, m=self.model.means_, s=self.model.covariances_, w=self.model.weights_, inv_s=self.inv_s, det_s=self.det_s, new_s=self.new_s, B=self.B),  np.arange(len(self.chunk)).astype(int)):
    #             self.result[loop_indices[0] +idx] = x*ds_s +ds_m
    #         if j % (chunks//100) == 0:
    #             np.save("./results/" + fname, self.result)
    #         f = open("./logs/log " + fname[:-4] +".txt", 'a')
    #         f.write("finished predicting chunk {}, it took {:2f} min \n".format(j, (time.time()-st1)/60))
    #         f.close()
    #     print("finished predicting, it took {:2f} min".format( (time.time()-st)/60))

    #     np.save("./results/" + fname, self.result)


    #     # # print(self.__dict__)
    #     # for idx, x in pool.map(self._procedure,  np.arange(len(indices)).astype(int)):
    #     # # for idx, x in pool.map(self._procedure,  np.arange(indices).astype(int)):

    #     #         result[idx] = x*ds_s +ds_m

    #     # pool.map(partial(_procedure, x_train=X, m=self.means_, s=self.covariances_, w=self.weights_, inv_s=self.inv_s, det_s=self.det_s, new_s=self.new_s, B=self.B, output=result),  np.arange(len(indices)).astype(int))

    #     # for j in range(chunks):
    #     #     loop_indices = indices[int(j*len(X)/chunks):int((j+1)*len(X)/chunks)]
    #     #     st1 = time.time()
    #     #     chunk = X[loop_indices]


    #     #     # for idx, x in pool.map(partial(_procedure, x_train=chunk, m=self.means_, s=self.covariances_, w=self.weights_, inv_s=self.inv_s, det_s=self.det_s, new_s=self.new_s, B=self.B),  np.arange(len(chunk)).astype(int)):
    #     #     #     result[loop_indices[0] +idx] = x*ds_s +ds_m

    #     #     for idx, x in pool.map(self._procedure,  np.arange(len(chunk)).astype(int)):
    #     #         result[loop_indices[0] +idx] = x*ds_s +ds_m

    #     #     if j % (chunks//100) == 0:
    #     #         np.save("./" + fname, result)

    #     print("finished predicting, it took {:2f} min".format( (time.time()-st)/60))
    #     print(result)
    #     print( np.sqrt(np.mean(np.square(result - X[:,-1])[:limit])))
    #     # del result
    #     # np.save("./" + fname, result)
    #     return result

    # # @classmethod
    # def _procedure(cls, i):
    #     x_train=cls.X
    #     m=cls.means_
    #     s=cls.covariances_
    #     w=cls.weights_
    #     inv_s=cls.inv_s
    #     det_s=cls.det_s
    #     new_s=cls.new_s
    #     B=cls.B

    #     # m, s, w = self.model.means_, self.model.covariances_, self.model.weights_
    #     # x_train, inv_s, det_s, new_s = self.x_train, self.inv_s, self.det_s, self.new_s
    #     y = x_train[i]
    #     dy = (y[:-1]-m[:,:-1]).reshape(len(m), len(y)-1, 1)

    #     new_m = m[:,-1:].reshape(len(m),1,1) + B @ dy

    #     ex1 =  -.5 * dy.reshape(len(m), 1, len(y)-1) @ (inv_s @ dy.reshape(len(m), len(y)-1, 1))
    #     K = 1 / np.sqrt ( np.power(2*np.pi, len(m)) * det_s )

    #     new_w  = w.reshape(-1,1,1) * K.reshape(-1,1,1) * np.exp(ex1).reshape(-1,1,1)
    #     new_w /= np.sum(new_w)

    #     f = lambda y : -1*np.sum( gmm_pdf(y, new_m[:,0], new_s[:,0,0], new_w[:,0] ))
    #     # return minimize(f, np.median(y[:-1]), method='Powell').x
    #     # cls.output[i] = minimize(f, np.median(y[:-1]), method='Powell').x
    #     return i, minimize(f, np.median(y[:-1]), method='Powell').x


def mulnormpdf(X, MU, SIGMA):
    """Evaluates the PDF for the multivariate Guassian distribution.
    Parameters
    ----------
    X : np array
        Inputs/entries row-wise. Can also be a 1-d array if only a 
        single point is evaluated.
    MU : nparray
        Center/mean, 1d array. 
    SIGMA : 2d np array
        Covariance matrix.
    Returns
    -------
    prob : 1d np array
        Probabilities for entries in `X`.
    
    Examplesp.data[data_names[-1]].values
    --------
    ::
        from pypr.clustering import *
        from numpy import *
        X = array([[0,0],[1,1]])
        MU = array([0,0])
        SIGMA = diag((1,1))
        gmm.mulnormpdf(X, MU, SIGMA)
    """
    # Check if inputs are ok:
    if MU.ndim != 1:
        raise (ValueError, "MU must be a 1 dimensional array")
    
    # Evaluate pdf at points or point:p.data[data_names[-1]].values
    mu = MU
    x = X.T
    if x.ndim == 1:
        x = np.atleast_2d(x).T
    sigma = np.atleast_2d(SIGMA) # So we also can use it for 1-d distributions

    N = len(MU)
    ex1 = np.dot(np.linalg.inv(sigma), (x.T-mu).T)
    ex = -0.5 * (x.T-mu).T * ex1
    if ex.ndim == 2: ex = np.sum(ex, axis = 0)
    K = 1 / np.sqrt ( np.power(2*np.pi, N) * np.linalg.det(sigma) )
    return K*np.exp(ex)

def gmm_pdf(X, centroids, ccov, mc, individual=False):
    """Evaluates the PDF for the multivariate Guassian mixture.
    Draw samples from a Mixture of Gaussians (MoG)
    Parameters
    ----------
    centroids : list
        List of cluster centers - [ [x1,y1,..],..,[xN, yN,..] ]
    ccov : list
        List of cluster co-variances DxD matrices
    mc : list
        Mixing cofficients for each cluster (must sum to one)
                  by default equal for each cluster.
    individual : bool
        If True the probability density is returned for each cluster component.
    Returns
    -------
    prob : 1d np array
        Probability density values for entries in `X`.
    """
    if individual:
        pdf = np.zeros((len(X), len(centroids)))
        for i in range(len(centroids)):
            pdf[:,i] = mulnormpdf(X, centroids[i], ccov[i]) * mc[i]
        return pdf
    else:
        pdf = None
        for i in range(len(centroids)):
            pdfadd = mulnormpdf(X, centroids[i], ccov[i]) * mc[i]
            if pdf==None:
                pdf = pdfadd
            else:
                pdf = pdf + pdfadd
        return pdf



# test = gmm(whiten=True)
# test.fit([[1,2,3,4],[1,2,3,4]])
# print(test.means_, test.covariances_)
# test.save_model('test_model')
# test.load_model('test_model')

# print(test.means_, test.covariances_)
