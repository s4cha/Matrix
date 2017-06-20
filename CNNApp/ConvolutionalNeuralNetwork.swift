//
//  ConvolutionalNeuralNetwork.swift
//  CNNApp
//
//  Created by Sacha Durand Saint Omer on 20/06/2017.
//  Copyright © 2017 freshOS. All rights reserved.
//

import Foundation

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
            newMatrix = l.runOn(matrix: newMatrix)
            print(newMatrix)
        }
        
        for l in layers {
            print(l)
            newMatrix = l.runOn(matrix: newMatrix)
            print(newMatrix)
        }
        
        
        return false
    }
}


