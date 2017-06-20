//
//  ViewController.swift
//  CNNApp
//
//  Created by Sacha Durand Saint Omer on 19/06/2017.
//  Copyright © 2017 freshOS. All rights reserved.
//

import UIKit

struct Params {
    
    struct Convolution {
        static let numberOfFeatures = 1
        static let sizeOfOfFeatures = 3
    }
    
    struct Pooling {
        static let windowSize = 2
        static let stride = 2
    }
    
    struct FullyConnected {
        static let numberOfNeurons = 3
    }
}



class ViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        run()
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
        
        let matrix: Matrix<Float> = [
            [1,0,0,0,1],
            [0,1,0,1,0],
            [0,0,1,0,0],
            [0,1,0,1,0],
            [1,0,0,0,1]
        ]
        
        let minusMatrix = replace(value: 0, by: -1, in: matrix)
        
        let cnn = ConvolutionalNeuralNetwork()
        print(cnn.isCross(minusMatrix))
    }

}

class ConvolutionalNeuralNetwork {
    func isCross(_ matrix: Matrix<Float>) -> Bool {
        
        let layers: [Layer] = [
            ConvolutionLayer(),
            RectifiedLinearUnitLayer(),
            PoolingLayer()
        ]
        
        var newMatrix = matrix
        
        print("Strat Matrix: \(matrix)")
        for l in layers {
            print(l)
            newMatrix = l.runOn(matrix: matrix)
            print(newMatrix)
        }
        
        
        return false
    }
}




