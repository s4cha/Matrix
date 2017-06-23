//
//  AppDelegate.swift
//  CNNApp
//
//  Created by Sacha Durand Saint Omer on 19/06/2017.
//  Copyright © 2017 freshOS. All rights reserved.
//

import UIKit


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

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {

    var window: UIWindow?
    
    func exp(_ matrix:Matrix<Float>) -> Matrix<Float> {
        var m = matrix
        for i in 0..<m.numberOfLines {
            for j in 0..<m.numberOfColomns {
                m[i, j] = pow(kEuler,  matrix[i, j])
            }
        }
        return m
    }
    
    func nonlin(x:Matrix<Float>, deriv:Bool = false) -> Matrix<Float> {
        if deriv {
            return x*(1-x)
        }
        return 1 / (1 + exp(-x))
    }
    
    func matrixMultiplication(m1:Matrix<Float>, m2:Matrix<Float>) throws -> Matrix<Float> {
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


    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplicationLaunchOptionsKey: Any]?) -> Bool {
        
        // Input dataset
        let X: Matrix<Float> = [
            [0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]
        ]
        
        // output dataset
        let Y: Matrix<Float> = [
            [0],
            [0],
            [1],
            [1]
        ]
       
        // seed random numbers to make calculation
        // deterministic (just a good practice)
//        np.random.seed(1)
//        
        // initialize weights randomly with mean 0
        //2*np.random.random((3,1)) - 1
        var syn0:Matrix<Float> = [
            [-0.16595599],
            [ 0.44064899],
            [-0.99977125]
        ]
        
        var l1:Matrix<Float>!
        for _ in 0...10 {
            // forward propagation
            let l0 = X
            
            let product = try! matrixMultiplication(m1: l0, m2: syn0)
            l1 = nonlin(x: product)
            print(l1)
            // how much did we miss?
            let l1_error = try! Y - l1
    
            // multiply how much we missed by the
            // slope of the sigmoid at the values in l1
            let l1_delta = l1_error * nonlin(x: l1,deriv: true)
            print(l1_delta)
            
            // update weights
            syn0 = try! syn0 + matrixMultiplication(m1:l0.transpose, m2:l1_delta)
        }
//        
        print("Output After Training:")
        print(l1)

        
        
//        run()
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
            ,
            [[0,0,1],
             [0,0,0],
             [0,0,0]]
            ,
            [[0,0,0],
             [0,0,0],
             [1,0,0]]
            ,
            [[1,1,1],
             [0,0,0],
             [1,0,1]]
            ,
            [[0,0,0],
             [0,0,0],
             [1,1,1]]
//            ,
//            [[0,0,0],
//             [0,0,0],
//             [1,0,0]]
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

