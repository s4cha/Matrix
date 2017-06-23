//
//  Matrix+Helpers.swift
//  CNNApp
//
//  Created by Sacha Durand Saint Omer on 19/06/2017.
//  Copyright © 2017 freshOS. All rights reserved.
//

import Foundation

func randomMatrix(ofShape: (Int, Int)) -> Matrix<Float> {
    var m = Matrix<Float>(ofShape: ofShape, with: 0)
    var t = 0
    for i in 0..<m.numberOfLines {
        for j in 0..<m.numberOfColomns {
            let rand:Float = Float(arc4random_uniform(10000)) / 10000.0
            m[i, j] = rand
        }
    }
    return m
}

func flatten(_ matrix: Matrix<Float>) -> Matrix<Float> {
    var m = Matrix<Float>(ofShape: (1, matrix.shape.0 * matrix.shape.1), with: 0)
    var t = 0
    for i in 0..<matrix.numberOfLines {
        for j in 0..<matrix.numberOfColomns {
            m[0, t] = matrix[i, j]
            t += 1
        }
    }
    return m
}

func exp(_ matrix:Matrix<Float>) -> Matrix<Float> {
    var m = matrix
    for i in 0..<m.numberOfLines {
        for j in 0..<m.numberOfColomns {
            m[i, j] = pow(kEuler,  matrix[i, j])
        }
    }
    return m
}

func nonlin(_ x:Matrix<Float>, deriv:Bool = false) -> Matrix<Float> {
    if deriv {
        return x*(1-x)
    }
    return 1 / (1 + exp(-x))
}

func dot(m1:Matrix<Float>, m2:Matrix<Float>) throws -> Matrix<Float> {
    if m1.shape.1 != m2.shape.0 {
        throw MatrixError.defaultError
    }
    var newMatrix:Matrix<Float> = Matrix(ofShape:(m1.shape.0, m2.shape.1), with:0)
    for i in 0..<newMatrix.numberOfLines {
        for j in 0..<newMatrix.numberOfColomns {
            var k:Float = 0
            for c in 0..<m1.numberOfColomns {
                k += m1[i, c] * m2[c, j]
            }
            newMatrix[i, j] = k
        }
    }
    return newMatrix
}

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
    for i in 0..<newMatrix.numberOfLines {
        for j in 0..<newMatrix.numberOfColomns {
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

let kEuler:Float = 2.71828182846

extension Matrix where T == Float {
    var transpose:Matrix<Float> {
        var m = Matrix(ofShape: (shape.1, shape.0), with: 0)
        for i in 0..<m.numberOfLines {
            for j in 0..<m.numberOfColomns {
                m[i, j] = self[j, i]
            }
        }
        return m
    }
}

extension Matrix {
    var shape:(Int, Int) {
        return (numberOfLines, numberOfColomns)
    }
}

func - (l:Float, r:Matrix<Float>) -> Matrix<Float> {
    var m = r
    for i in 0..<m.numberOfLines {
        for j in 0..<m.numberOfColomns {
            m[i, j] = l - m[i, j]
        }
    }
    return m
}

func - (l:Matrix<Float>, r:Float) -> Matrix<Float> {
    var m = l
    for i in 0..<m.numberOfLines {
        for j in 0..<m.numberOfColomns {
            m[i, j] = m[i, j] - r
        }
    }
    return m
}

func + (l:Float, r:Matrix<Float>) -> Matrix<Float> {
    var m = r
    for i in 0..<m.numberOfLines {
        for j in 0..<m.numberOfColomns {
            m[i, j] = l + m[i, j]
        }
    }
    return m
}

func / (l:Float, r:Matrix<Float>) -> Matrix<Float> {
    var m = r
    for i in 0..<m.numberOfLines {
        for j in 0..<m.numberOfColomns {
            m[i, j] = l / m[i, j]
        }
    }
    return m
}

func * (l:Float, r:Matrix<Float>) -> Matrix<Float> {
    var m = r
    for i in 0..<m.numberOfLines {
        for j in 0..<m.numberOfColomns {
            m[i, j] = l * m[i, j]
        }
    }
    return m
}


prefix func - (r:Matrix<Float>) -> Matrix<Float> {
    var m = r
    for i in 0..<m.numberOfLines {
        for j in 0..<m.numberOfColomns {
            m[i, j] = -m[i, j]
        }
    }
    return m
}

func - (l:Matrix<Float>, r:Matrix<Float>) throws -> Matrix<Float> {
    if l.shape != r.shape {
        throw MatrixError.defaultError
    }
    var m = r
    for i in 0..<m.numberOfLines {
        for j in 0..<m.numberOfColomns {
            m[i, j] = l[i, j] - r[i, j]
        }
    }
    return m
}

func + (l:Matrix<Float>, r:Matrix<Float>) throws -> Matrix<Float> {
    if l.shape != r.shape {
        throw MatrixError.defaultError
    }
    var m = r
    for i in 0..<m.numberOfLines {
        for j in 0..<m.numberOfColomns {
            m[i, j] = l[i, j] + r[i, j]
        }
    }
    return m
}

func * (l:Matrix<Float>, r:Matrix<Float>) -> Matrix<Float> {
    return multipy(l, with: r)
}

enum MatrixError: Error {
    case defaultError
}
