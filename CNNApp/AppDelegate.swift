//
//  AppDelegate.swift
//  CNNApp
//
//  Created by Sacha Durand Saint Omer on 19/06/2017.
//  Copyright © 2017 freshOS. All rights reserved.
//

import UIKit

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {

    var window: UIWindow?
    
    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplicationLaunchOptionsKey: Any]?) -> Bool {
        
        let randomNumber = arc4random()
        print(randomNumber)
        
        
        let twoLayersNN = TwoLayerNeuralNet()
        twoLayersNN.run5()
        
        
        // Input dataset
        var X: Matrix<Float> = [
            [1,0,0,1], // is Slash
            [0,0,0,0],
            
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
            
            [1,1,0,0],
            [1,0,1,0],
//            [1,0,0,1],
            [0,1,1,0],
            [0,1,0,1],
            [0,0,1,1],
            
            [0,1,1,1],
            [1,0,1,1],
            [1,1,0,1],
            [1,1,1,0],
            
            [1,1,1,1],
        ]
        
//        print(X.shape)
        
//        X = replace(value: 0, by: -1, in: X)
        
        
        // output dataset
        let Y: Matrix<Float> = [
            [1],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0]
        ]
        
        // seed random numbers to make calculation
        // deterministic (just a good practice)
        //        np.random.seed(1)
        //
        // initialize weights randomly with mean 0
        //2*np.random.random((3,1)) - 1
        var syn0:Matrix<Float> = [
            [-0.16595599],
            [ 0.44064899],
            [ 0.44064899],
            [-0.99977125]
        ]
        
        
        var l1:Matrix<Float>!
        for i in 0...0 {
            
            // forward propagation
            let l0 = X
            
            let product = try! dot(m1: l0, m2: syn0)
            l1 = nonlin(product)
            
            if i % 100 == 0 {
                
            }
            // how much did we miss?
            let l1_error = try! Y - l1
            
            // multiply how much we missed by the
            // slope of the sigmoid at the values in l1
            let l1_delta = l1_error * nonlin(l1,deriv: true)
            
            // update weights
            syn0 = try! syn0 + dot(m1:l0.transpose, m2:l1_delta)
        }
        //
        print("Output After Training:")
        print(l1)
        
        
        
        print(syn0)
        // Trained weights
        let trainedWeights = syn0
        // Input dataset
        let inputToPredict: Matrix<Float> = [
            [1,0,0,1], // is Slash
        ]
        
        let product = try! dot(m1: inputToPredict, m2: trainedWeights)
        let prediction = nonlin(product)
        print("prediction : \(prediction)")
        if prediction[0, 0] > 0.8 {
            print("This is a slash")
        } else {
            print("This is not a slash")
        }
        
        
//        // Input dataset
//        let X: Matrix<Float> = [
//            [0,0,1],
//            [0,1,1],
//            [1,0,1],
//            [1,1,1]
//        ]
//        
//        // output dataset
//        let Y: Matrix<Float> = [
//            [0],
//            [0],
//            [1],
//            [1]
//        ]
//       
//        // seed random numbers to make calculation
//        // deterministic (just a good practice)
////        np.random.seed(1)
////        
//        // initialize weights randomly with mean 0
//        //2*np.random.random((3,1)) - 1
//        var syn0:Matrix<Float> = [
//            [-0.16595599],
//            [ 0.44064899],
//            [-0.99977125]
//        ]
//        
//        var l1:Matrix<Float>!
//        for _ in 0...10 {
//            // forward propagation
//            let l0 = X
//            
//            let product = try! matrixMultiplication(m1: l0, m2: syn0)
//            l1 = nonlin(x: product)
//            print(l1)
//            // how much did we miss?
//            let l1_error = try! Y - l1
//    
//            // multiply how much we missed by the
//            // slope of the sigmoid at the values in l1
//            let l1_delta = l1_error * nonlin(x: l1,deriv: true)
//            print(l1_delta)
//            
//            // update weights
//            syn0 = try! syn0 + matrixMultiplication(m1:l0.transpose, m2:l1_delta)
//        }
////        
//        print("Output After Training:")
//        print(l1)

        
        
//        run()
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

