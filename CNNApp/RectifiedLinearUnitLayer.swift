//
//  RectifiedLinearUnitLayer.swift
//  CNNApp
//
//  Created by Sacha Durand Saint Omer on 19/06/2017.
//  Copyright © 2017 freshOS. All rights reserved.
//

import Foundation

// Keeps the math right by just setting negative values to 0
class RectifiedLinearUnitLayer: Layer {
    
    func runOn(matrix: Matrix<Float>) -> Matrix<Float> {
        var newMatrix = matrix
        for i in 0..<matrix.backing.count {
            for j in 0..<matrix.backing.count {
                if matrix[i,j] < 0 {
                    newMatrix[i,j] = 0
                }
            }
        }
        return newMatrix
    }
}
