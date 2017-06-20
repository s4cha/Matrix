//
//  ConvolutionLayer.swift
//  CNNApp
//
//  Created by Sacha Durand Saint Omer on 20/06/2017.
//  Copyright © 2017 freshOS. All rights reserved.
//

import Foundation

// MAtch pieces of the images instead
// Filtering
// use floats
// move filter around, compute and crate a map
//Repeated application of filter = CONVOLUTION


//Filtering
// Line Up with the image
// Multiply pixel by feature pixel
// Add them up
// Divide by number of pixels in the feature
// Build a map
// repeat with every features

class ConvolutionLayer: Layer {
    
    func runOn(matrix: Matrix<Float>) -> Matrix<Float> {
        
        // Todo find feature alone
        let feature: Matrix<Float> = [
            [1,0,0],
            [0,1,0],
            [0,0,1]
        ]
//        let newMatrix = replace(value: 0, by: -1, in: matrix)
//        print("newMatrix")
//        print(newMatrix)
//        
//        let test = multipy(feature, with:factor)
//        print("Result multiplication")
//        print(test)
    
        
        let convoluted: Matrix<Float> = [
            [0.77,-12,0,0,1],
            [-0.11,1,0.33,-0.11, 0],
            [0,-3, 1,-34,0],
            [0,-0.345,-0.4,1,0],
            [0.33,-0.11,-0.4,0,0.77]
        ]
        return convoluted
    }
}
