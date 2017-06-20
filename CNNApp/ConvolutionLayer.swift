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
        
        var convoluted = fill(matrix, with:0)
        print(convoluted)
        
        // Todo find feature alone
        let feature: Matrix<Float> = [
            [1,-1,-1],
            [-1,1,-1],
            [-1,-1,1]
        ]
        
        // Feature needs to be ood, feature
        
        for i in 0..<matrix.backing.count {
            for j in 0..<matrix.backing.count {
                
                var subMatrix: Matrix<Float> = [
                    [0,0,0],
                    [0,0,0],
                    [0,0,0]
                ]
                
                for fi in 0..<feature.backing.count {
                    for fj in 0..<feature.backing.count {
                        let ci = fi - 1 + i
                        let cj = fj - 1 + j
                        
                        var value:Float = 0
                        if ci >= 0 && cj >= 0
                            && ci < matrix.backing.count && cj < matrix.backing.count {
                            value = matrix[ci, cj]
                        }
                        subMatrix[fi,fj] = value
                        let aVal = matrix[i, j]
                    }
                }
                
                let multiplied = multipy(subMatrix, with:feature)
                let mean = meanOf(multiplied)
                convoluted[i, j] = mean
            }
        }
        return convoluted
    }
}
