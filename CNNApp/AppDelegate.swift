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

