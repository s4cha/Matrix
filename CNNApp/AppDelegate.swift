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
        
//        let matrix: Matrix<Float> = [
//            [1,0,1],
//            [0,1,0],
//            [1,0,1]
//        ]
       
//        let matrix: Matrix<Float> = [
//            [1,0,0,0,1],
//            [0,1,0,1,0],
//            [0,0,1,0,0],
//            [0,1,0,1,0],
//            [1,0,0,0,1]
//        ]
        
//        let matrix: Matrix<Float> = [
//            [0,0,0,0,0,0,0],
//            [0,1,0,0,0,1,0],
//            [0,0,1,0,1,0,0],
//            [0,0,0,1,0,0,0],
//            [0,0,1,0,1,0,0],
//            [0,1,0,0,0,1,0],
//            [0,0,0,0,0,0,0],
//        ]
        
        let matrix: Matrix<Float> = [
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,1,0,0],
            [0,0,0,1,0,0,0],
            [0,0,1,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ]
        
        let minusMatrix = replace(value: 0, by: -1, in: matrix)
        
        let cnn = ConvolutionalNeuralNetwork()
        print(cnn.isCross(minusMatrix))
    }

}

