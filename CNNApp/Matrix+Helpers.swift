//
//  Matrix+Helpers.swift
//  CNNApp
//
//  Created by Sacha Durand Saint Omer on 19/06/2017.
//  Copyright © 2017 freshOS. All rights reserved.
//

import Foundation

func multipy(_ matrix:Matrix<Float>, with:Matrix<Float>) -> Matrix<Float> {
    var newMatrix = matrix
    for i in 0..<matrix.numberOfLines {
        for j in 0..<matrix.numberOfColomns {
            let v = matrix[i,j]
            let v2 = with[i,j]
            newMatrix[i,j] = v*v2
        }
    }
    return newMatrix
}

func replace<T: Equatable>(value:T, by:T, in matrix:Matrix<T>) -> Matrix<T> {
    var newMatrix = matrix
    for i in 0..<newMatrix.backing.count {
        for j in 0..<newMatrix.backing.count {
            let v = newMatrix[i,j]
            if v == value {
                newMatrix[i,j] = by
            }
        }
    }
    return newMatrix
}

func maxOf(_ matrix:Matrix<Float>) -> Float {
    var max:Float = -100
    for i in 0..<matrix.backing.count {
        for j in 0..<matrix.backing.count {
            let v = matrix[i,j]
            if v > max {
                max = v
            }
        }
    }
    return max
}

func sumOf(_ matrix:Matrix<Float>) -> Float {
    var sum:Float = 0
    for i in 0..<matrix.backing.count {
        for j in 0..<matrix.backing.count {
            let v = matrix[i,j]
            sum += v
        }
    }
    return sum
}

func meanOf(_ matrix:Matrix<Float>) -> Float {
    return sumOf(matrix) / Float(numberOfEntries(matrix))
}

func numberOfEntries(_ matrix:Matrix<Float>) -> Int {
    return matrix.backing.count * matrix.backing.count
}

func fill(_ matrix:Matrix<Float>, with:Float) -> Matrix<Float> {
    var newMatrix = matrix
    for i in 0..<matrix.backing.count {
        for j in 0..<matrix.backing.count {
            newMatrix[i,j] = with
        }
    }
    return newMatrix
}


func subMatrix(ofSize:Int, from matrix: Matrix<Float>, atX:Int, y:Int) -> Matrix<Float> {
    
    // Matrix of 3.
    var subMatrix: Matrix<Float> = [
        [-13,-13],
        [-13,-13]
    ]
    
    var si = 0
    for i in atX..<atX+ofSize {
        var sj = 0
        for j in y..<y+ofSize {
            if j < matrix.backing.count && i < matrix.backing.count {
                let v = matrix[i,j]
                subMatrix[si,sj] = v
            }
            sj += 1
        }
        si += 1
    }
    
    return subMatrix
}
