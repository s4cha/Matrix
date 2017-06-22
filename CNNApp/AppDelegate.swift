//
//  AppDelegate.swift
//  CNNApp
//
//  Created by Sacha Durand Saint Omer on 19/06/2017.
//  Copyright © 2017 freshOS. All rights reserved.
//

import UIKit


extension Matrix {
    var shape:(Int, Int) {
        return (numberOfLines, numberOfColomns)
    }
}

enum MatrixError: Error {
    case defaultError
}

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {

    var window: UIWindow?
    
//    
    func matrixMultiplication(m1:Matrix<Float>, m2:Matrix<Float>) throws -> Matrix<Float> {
        if m1.shape.1 != m2.shape.0 {
            throw MatrixError.defaultError
        }
        var newMatrix:Matrix<Float> = Matrix(ofShape:(m1.shape.0, m2.shape.1), with:0)
        for i in 0..<newMatrix.numberOfLines {
            for j in 0..<newMatrix.numberOfColomns {
                var k:Float = 0
                for c in 0..<m1.numberOfColomns {
                    k += m1[i, c] * m2[c, j]
                }
                newMatrix[i, j] = k
            }
        }
        return newMatrix
    }


    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplicationLaunchOptionsKey: Any]?) -> Bool {
        
        
        
        let t1:Matrix<Float> = [
            [5, 1],
            [2, 3],
            [3, 4]
        ]
        
        let t2:Matrix<Float> = [
            [1,2,0],
            [4,3,-1]
        ]
        
        let multiplied = try! matrixMultiplication(m1:t1, m2: t2)
        print(multiplied.shape)
        print(multiplied)
        
    
        
//        # sigmoid function
//        def nonlin(x,deriv=False):
//        if(deriv==True):
//        return x*(1-x)
//        return 1/(1+np.exp(-x))
        
        // Input dataset
        let X: Matrix<Float> = [
            [0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]
        ]
        
        print(X.shape)
        
//        // output dataset
//        let Y: Matrix<Float> = [
//            [0],
//            [0],
//            [1],
//            [1]
//        ]
        
        let Y: Matrix<Float> = [
            [0, 0, 1, 1],
        ]
        
        print(Y.shape)
       
        // seed random numbers to make calculation
        // deterministic (just a good practice)
//        np.random.seed(1)
//        
        // initialize weights randomly with mean 0
        //2*np.random.random((3,1)) - 1
        let syn0 = [
            [-0.16595599],
            [ 0.44064899],
            [-0.99977125]
        ]
        
        for i in 0...1000 {
            // forward propagation
            let l0 = X
    //        l1 = nonlin(np.dot(l0,syn0))
    //        
    //        # how much did we miss?
    //        l1_error = y - l1
    //        
    //        # multiply how much we missed by the
    //        # slope of the sigmoid at the values in l1
    //        l1_delta = l1_error * nonlin(l1,True)
    //        
    //        # update weights
    //        syn0 += np.dot(l0.T,l1_delta)
            }
//        
//        print "Output After Training:"
//        print l1

        
        
        run()
        return true
    }


    func run() {
        print("PARAMS")
        print("  Convolution")
        print("    numberOfFeatures: \(Params.Convolution.numberOfFeatures)")
        print("    sizeOfOfFeatures: \(Params.Convolution.sizeOfOfFeatures)")
        print("  Pooling")
        print("    windowSize: \(Params.Pooling.windowSize)")
        print("    stride: \(Params.Pooling.stride)")
        print("  FullyConnected")
        print("    numberOfNeurons: \(Params.FullyConnected.numberOfNeurons)")
        // Is image an X or an O ?
        // 1 = white
        // -1 / 0 = blakc pixel
        
        let xs:[Matrix<Float>] = [
//            [[1,0,0,0,1],
//             [0,1,0,1,0],
//             [0,0,1,0,0],
//             [0,1,0,1,0],
//             [1,0,0,0,1]]
            [[1,0,0],
             [0,1,0],
             [0,0,1]]
//            ,
//            [[0,0,0,0,1], // no topLeft corner
//             [0,1,0,1,0],
//             [0,0,1,0,0],
//             [0,1,0,1,0],
//             [1,0,0,0,1]]
//            ,
//            [[1,0,0,0,0], // no topRight corner
//             [0,1,0,1,0],
//             [0,0,1,0,0],
//             [0,1,0,1,0],
//             [1,0,0,0,1]]
//            ,
//            [[1,0,0,0,1], // no BottomRight corner
//             [0,1,0,1,0],
//             [0,0,1,0,0],
//             [0,1,0,1,0],
//             [1,0,0,0,0]]
//            ,
//            [[1,0,0,0,1], // no BottomLeft corner
//             [0,1,0,1,0],
//             [0,0,1,0,0],
//             [0,1,0,1,0],
//             [0,0,0,0,1]]
//            ,
//            [[1,0,0,0,0], // only TopLeft corner
//            [0,1,0,1,0],
//            [0,0,1,0,0],
//            [0,1,0,1,0],
//            [0,0,0,0,0]]
//            ,
//            [[0,0,0,0,1], // only TopRight corner
//            [0,1,0,1,0],
//            [0,0,1,0,0],
//            [0,1,0,1,0],
//            [0,0,0,0,0]]
//            ,
//            [[0,0,0,0,0], // only BottomLeft corner
//             [0,1,0,1,0],
//             [0,0,1,0,0],
//             [0,1,0,1,0],
//             [0,0,0,0,1]]
//            ,
//            [[0,0,0,0,0], // only BottomRight corner
//            [0,1,0,1,0],
//            [0,0,1,0,0],
//            [0,1,0,1,0],
//            [1,0,0,0,0]]
//            ,
//            [[0,0,0,0,0], // smaller
//             [0,1,0,1,0],
//             [0,0,1,0,0],
//             [0,1,0,1,0],
//             [0,0,0,0,0]]
//            ,
//            [[0,0,0,0,0], // No Top
//             [0,1,0,1,0],
//             [0,0,1,0,0],
//             [0,1,0,1,0],
//             [1,0,0,0,1]]
//            ,
//            [[1,0,0,0,0], // No Right
//            [0,1,0,1,0],
//            [0,0,1,0,0],
//            [0,1,0,1,0],
//            [1,0,0,0,0]]
//            ,
//            [[1,0,0,0,1], // No Bottom
//            [0,1,0,1,0],
//            [0,0,1,0,0],
//            [0,1,0,1,0],
//            [0,0,0,0,0]]
//            ,
//            [[0,0,0,0,1], // No Left
//            [0,1,0,1,0],
//            [0,0,1,0,0],
//            [0,1,0,1,0],
//            [0,0,0,0,1]]
//            ,
//            [[0,0,0,0,1], // No Diagonal 1
//             [0,1,0,1,0],
//             [0,0,1,0,0],
//             [0,1,0,1,0],
//             [1,0,0,0,0]]
//            ,
//            [[1,0,0,0,0], // No Diagonal 2
//             [0,1,0,1,0],
//             [0,0,1,0,0],
//             [0,1,0,1,0],
//             [0,0,0,0,1]]
        ]
        
        
        let notxs:[Matrix<Float>] = [
            [[0,0,0],
            [0,0,0],
            [0,0,0]]
            ,
            [[0,0,1],
             [0,0,0],
             [0,0,0]]
            ,
            [[0,0,0],
             [0,0,0],
             [1,0,0]]
            ,
            [[1,1,1],
             [0,0,0],
             [1,0,1]]
            ,
            [[0,0,0],
             [0,0,0],
             [1,1,1]]
//            ,
//            [[0,0,0],
//             [0,0,0],
//             [1,0,0]]
//            ,
//            [[1,0,0,0,0],
//             [0,0,0,0,0],
//             [0,0,0,0,0],
//             [0,0,0,0,0],
//             [0,0,0,0,0]]
//            ,
//            [[1,1,1,1,0],
//             [0,0,0,0,0],
//             [0,0,0,0,0],
//             [0,0,0,0,0],
//             [0,0,0,0,0]]
//            ,
//            [[1,0,0,0,0],
//             [0,1,0,0,0],
//             [0,0,1,0,0],
//             [0,0,0,0,0],
//             [0,0,0,0,0]]
//            ,
//            [[0,0,0,0,0],
//             [0,1,0,0,0],
//             [0,0,1,0,0],
//             [0,0,0,1,0],
//             [0,0,0,0,1]]
//            ,
//            [[0,0,0,0,0],
//             [1,0,0,0,1],
//             [1,0,0,0,1],
//             [0,0,0,0,1],
//             [0,0,0,0,1]]
//            ,
//            [[0,0,1,1,1],
//             [0,0,1,0,0],
//             [0,1,1,0,0],
//             [0,0,0,0,0],
//             [0,0,0,0,0]]
//            ,
//            [[0,0,1,1,1],
//             [0,0,1,0,0],
//             [0,1,0,0,0],
//             [0,0,0,0,0],
//             [0,0,0,0,1]]
//            ,
//            [[1,0,1,1,1],
//             [0,0,1,0,0],
//             [0,1,1,0,0],
//             [0,0,0,0,0],
//             [0,0,0,0,0]]
//            ,
//            [[1,0,1,1,1],
//             [1,0,1,0,0],
//             [1,1,0,0,0],
//             [1,0,0,0,0],
//             [1,1,1,1,1]]
//            ,
//            [[1,0,1,1,0],
//             [1,0,1,0,1],
//             [1,1,0,0,1],
//             [1,0,0,0,0],
//             [1,1,1,1,1]]
//            ,
//            [[1,0,1,1,0],
//             [1,0,0,0,1],
//             [1,0,0,0,1],
//             [1,0,0,0,0],
//             [1,1,1,1,1]]
        ]
        
        var trainingData = [(Matrix<Float>, [Float])]()
        for x in xs {
            trainingData.append((x,[1,0]))
        }
        for notx in notxs {
            trainingData.append((notx,[0,1]))
        }
        
        
        // Train CNN
        let cnn = ConvolutionalNeuralNetwork()
        cnn.train(with: trainingData)
    }

}

