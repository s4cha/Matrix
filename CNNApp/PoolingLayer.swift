//
//  PoolingLayer.swift
//  CNNApp
//
//  Created by Sacha Durand Saint Omer on 19/06/2017.
//  Copyright © 2017 freshOS. All rights reserved.
//

import Foundation

// Reduces the image size while keeping the feature
class PoolingLayer: Layer {
    
    func runOn(matrix: Matrix<Float>) -> Matrix<Float> {
        let windowSize = Params.Pooling.windowSize
        let stride = Params.Pooling.stride
        return runOn(matrix: matrix, windowSize: windowSize, stride: stride)
    }
    
    private func runOn(matrix: Matrix<Float>, windowSize:Int, stride: Int) -> Matrix<Float> {
        var size = matrix.backing.count
        if size % 2 != 0 {
            size += 1
        }
        size = size / 2
        
        var pooled = Matrix<Float>(size:size, with:0)
        
        var pi = 0
        var i = 0
        while i < size*windowSize {
            var j = 0
            var pj = 0
            while j < size*windowSize {
                let sm = subMatrix(ofSize: windowSize, from: matrix, atX: i, y: j)
                let max = maxOf(sm)
                pooled[pi,pj] = max
                j += stride
                pj += 1
            }
            i += stride
            pi += 1
        }
        return pooled
    }
}

