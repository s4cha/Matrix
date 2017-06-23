//
//  2LayerNeuralNet.swift
//  CNNApp
//
//  Created by Sacha Durand Saint Omer on 23/06/2017.
//  Copyright © 2017 freshOS. All rights reserved.
//

import Foundation

class TwoLayerNeuralNet {
    
    func run2pred() {
        
        // Input dataset
        let X: Matrix<Float> = [
            [0,0,0], //empty
            [0,1,1],
            [1,0,1],
            [1,1,1] // full
        ]
        
        // output dataset
        let Y: Matrix<Float> = [
            [0,1],
            [0,0],
            [0,0],
            [1,0]
        ]
        
        var syn0:Matrix<Float> =  2 * randomMatrix(ofShape: (3,4)) - 1
        var syn1:Matrix<Float> =  2 * randomMatrix(ofShape: (4,2)) - 1
        
        var l1:Matrix<Float>!
        var l2:Matrix<Float>!
        for j in 0...1000 { // try 60 000
            
            
            // Feed forward through layers 0, 1, and 2
            let l0 = X
            let product = try! dot(m1: l0, m2: syn0)
            l1 = nonlin(product)
            
            let product2 = try! dot(m1: l1, m2: syn1)
            l2 = nonlin(product2)
            
            // how much did we miss the target value?
            let l2_error = try! Y - l2
            
            if j % 10000 == 0 {
                print("error")
                //                print("Error:\((np.mean(np.abs(l2_error))))"
            }
            
            
            // in what direction is the target value?
            // were we really sure? if so, don't change too much.
            let l2_delta = l2_error * nonlin(l2, deriv:true)
            
            // how much did each l1 value contribute to the l2 error (according to the weights)?
            let l1_error = try! dot(m1: l2_delta, m2: syn1.transpose) //   l2_delta.dot(syn1.T)
            
            // in what direction is the target l1?
            // were we really sure? if so, don't change too much.
            let l1_delta = l1_error * nonlin(l1,deriv:true)
            
            syn1 = try! syn1 + dot(m1: l1.transpose, m2:l2_delta)  //+= l1.T.dot(l2_delta)
            syn0 = try! syn0 + dot(m1: l0.transpose, m2: l1_delta) //+= l0.T.dot(l1_delta)
        }
        //
        print("Output After Training:")
        print(l1)
        
        let trainedWeights0 = syn0
        let trainedWeights1 = syn1
        let inputToPredict: Matrix<Float> = [
            [1,1,1]
        ]
        
        let product = try! dot(m1: inputToPredict, m2: syn0)
        let l = nonlin(product)
        let product2 = try! dot(m1: l, m2: syn1)
        let prediction = nonlin(product2)
        
        print("prediction : \(prediction)")
        if prediction[0, 0] > 0.8 {
            print("This is line")
        } else if prediction[0, 1] > 0.8 {
            print("This is empty")
        } else {
            print("This isnothing")
        }
    }
    
    
    func run() {
        
        
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
            [1],
            [1],
            [0]
        ]
        
        // seed random numbers to make calculation
        // deterministic (just a good practice)
        //        np.random.seed(1)
        //
        // initialize weights randomly with mean 0
        //2*np.random.random((3,1)) - 1
        var syn0:Matrix<Float> =  2 * randomMatrix(ofShape: (3,4)) - 1
        var syn1:Matrix<Float> =  2 * randomMatrix(ofShape: (4,1)) - 1

        var l1:Matrix<Float>!
        var l2:Matrix<Float>!
        for j in 0...1000 { // try 60 000
            
            
            // Feed forward through layers 0, 1, and 2
            let l0 = X
            let product = try! dot(m1: l0, m2: syn0)
            l1 = nonlin(product)
            
            let product2 = try! dot(m1: l1, m2: syn1)
            l2 = nonlin(product2)
        
            // how much did we miss the target value?
            let l2_error = try! Y - l2
     
            if j % 10000 == 0 {
                print("error")
//                print("Error:\((np.mean(np.abs(l2_error))))"
            }
            
            
            // in what direction is the target value?
            // were we really sure? if so, don't change too much.
            let l2_delta = l2_error * nonlin(l2, deriv:true)
        
            // how much did each l1 value contribute to the l2 error (according to the weights)?
            let l1_error = try! dot(m1: l2_delta, m2: syn1.transpose) //   l2_delta.dot(syn1.T)
            
            // in what direction is the target l1?
            // were we really sure? if so, don't change too much.
            let l1_delta = l1_error * nonlin(l1,deriv:true)
            
            syn1 = try! syn1 + dot(m1: l1.transpose, m2:l2_delta)  //+= l1.T.dot(l2_delta)
            syn0 = try! syn0 + dot(m1: l0.transpose, m2: l1_delta) //+= l0.T.dot(l1_delta)
            
            
//            // how much did we miss?
//            let l1_error = try! Y - l1
//            
//            // multiply how much we missed by the
//            // slope of the sigmoid at the values in l1
//            let l1_delta = l1_error * nonlin(x: l1,deriv: true)
//            
//            // update weights
//            syn0 = try! syn0 + dot(m1:l0.transpose, m2:l1_delta)
        }
        //
        print("Output After Training:")
        print(l1)
        

        
    
        let trainedWeights0 = syn0
        let trainedWeights1 = syn1
        let inputToPredict: Matrix<Float> = [
            [1,0,1]
        ]
        
  
        let product = try! dot(m1: inputToPredict, m2: syn0)
        let l = nonlin(product)
        let product2 = try! dot(m1: l, m2: syn1)
        let prediction = nonlin(product2)

        print("prediction : \(prediction)")
        if prediction[0, 0] > 0.8 {
            print("This is a slash")
        } else {
            print("This is not a slash")
        }



        
    }
    
    func run2() {
        
        // Input dataset
        var X: Matrix<Float> = [
            [1,0,0,1], // is Slash
            [0,0,0,0],
            
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
            
            [1,1,0,0],
            [1,0,1,0],
            //            [1,0,0,1],
            [0,1,1,0],
            [0,1,0,1],
            [0,0,1,1],
            
            [0,1,1,1],
            [1,0,1,1],
            [1,1,0,1],
            [1,1,1,0],
            
            [1,1,1,1],
            ]
        
        // output dataset
        let Y: Matrix<Float> = [
            [1],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0]
        ]
        
        // seed random numbers to make calculation
        // deterministic (just a good practice)
        //        np.random.seed(1)
        //
        // initialize weights randomly with mean 0
        //2*np.random.random((3,1)) - 1

        
        var syn0:Matrix<Float> =  2 * randomMatrix(ofShape: (4,4)) - 1
        var syn1:Matrix<Float> =  2 * randomMatrix(ofShape: (4,1)) - 1
        
        var l1:Matrix<Float>!
        var l2:Matrix<Float>!
        for j in 0...10 { // try 60 000
            
            
            // Feed forward through layers 0, 1, and 2
            let l0 = X
            let product = try! dot(m1: l0, m2: syn0)
            l1 = nonlin(product)
            
            let product2 = try! dot(m1: l1, m2: syn1)
            l2 = nonlin(product2)
            
            // how much did we miss the target value?
            let l2_error = try! Y - l2
            
            if j % 10000 == 0 {
                print("error")
                //                print("Error:\((np.mean(np.abs(l2_error))))"
            }
            
            
            // in what direction is the target value?
            // were we really sure? if so, don't change too much.
            let l2_delta = l2_error * nonlin(l2, deriv:true)
            
            // how much did each l1 value contribute to the l2 error (according to the weights)?
            let l1_error = try! dot(m1: l2_delta, m2: syn1.transpose) //   l2_delta.dot(syn1.T)
            
            // in what direction is the target l1?
            // were we really sure? if so, don't change too much.
            let l1_delta = l1_error * nonlin(l1,deriv:true)
            
            syn1 = try! syn1 + dot(m1: l1.transpose, m2:l2_delta)  //+= l1.T.dot(l2_delta)
            syn0 = try! syn0 + dot(m1: l0.transpose, m2: l1_delta) //+= l0.T.dot(l1_delta)
            
            
            //            // how much did we miss?
            //            let l1_error = try! Y - l1
            //
            //            // multiply how much we missed by the
            //            // slope of the sigmoid at the values in l1
            //            let l1_delta = l1_error * nonlin(x: l1,deriv: true)
            //
            //            // update weights
            //            syn0 = try! syn0 + dot(m1:l0.transpose, m2:l1_delta)
        }
        //
        print("Output After Training:")
        print(l1)
        
        
        
        
        let trainedWeights0 = syn0
        let trainedWeights1 = syn1
        let inputToPredict: Matrix<Float> = [
            [1,0,0,1]
        ]
        
        
        let product = try! dot(m1: inputToPredict, m2: syn0)
        let l = nonlin(product)
        let product2 = try! dot(m1: l, m2: syn1)
        let prediction = nonlin(product2)
        
        print("prediction : \(prediction)")
        if prediction[0, 0] > 0.8 {
            print("This is a slash")
        } else {
            print("This is not a slash")
        }
        
        

    }
    
    //////////
    
    
    
    
    

    
    func run3() {
        
        
        let xs: [Matrix<Float>] = [
            [[1,0,1],
            [0,1,0],
            [1,0,1]] // X
            ,
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
            ,
            [[0,0,1],
             [0,1,0],
             [1,0,1]]
        ]
        
        // Transform 2d X in 1d X
        // AKA transofrm
        // [1,0,1],
        // [0,1,0],
        // [1,0,1]]
        // into
        // [1,0,1,0,1,0,1,0,1] // Do we loose spatial info?
        let flattened = xs.map { flatten($0) }
        var X: Matrix<Float> = Matrix(ofShape: (7,9), with: -1)
        for i in 0..<X.numberOfLines {
            for j in 0..<X.numberOfColomns {
                X[i, j] = flattened[i][0,j]
                print(X[i, j])
            }
        }
        
        // output dataset
        let Y: Matrix<Float> = [
            [1],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0]
        ]
        
        // seed random numbers to make calculation
        // deterministic (just a good practice)
        //        np.random.seed(1)
        //
        // initialize weights randomly with mean 0
        //2*np.random.random((3,1)) - 1
        var syn0:Matrix<Float> =  2 * randomMatrix(ofShape: (9,9)) - 1
        var syn1:Matrix<Float> =  2 * randomMatrix(ofShape: (9,1)) - 1
        
        var l1:Matrix<Float>!
        var l2:Matrix<Float>!
        for j in 0...1000 { // try 60 000
            
            
            // Feed forward through layers 0, 1, and 2
            let l0 = X
            let product = try! dot(m1: l0, m2: syn0)
            l1 = nonlin(product)
            
            let product2 = try! dot(m1: l1, m2: syn1)
            l2 = nonlin(product2)
            
            // how much did we miss the target value?
            let l2_error = try! Y - l2
            
            if j % 10000 == 0 {
                print("error")
                //                print("Error:\((np.mean(np.abs(l2_error))))"
            }
            
            
            // in what direction is the target value?
            // were we really sure? if so, don't change too much.
            let l2_delta = l2_error * nonlin(l2, deriv:true)
            
            // how much did each l1 value contribute to the l2 error (according to the weights)?
            let l1_error = try! dot(m1: l2_delta, m2: syn1.transpose) //   l2_delta.dot(syn1.T)
            
            // in what direction is the target l1?
            // were we really sure? if so, don't change too much.
            let l1_delta = l1_error * nonlin(l1,deriv:true)
            
            syn1 = try! syn1 + dot(m1: l1.transpose, m2:l2_delta)  //+= l1.T.dot(l2_delta)
            syn0 = try! syn0 + dot(m1: l0.transpose, m2: l1_delta) //+= l0.T.dot(l1_delta)
            
            
            //            // how much did we miss?
            //            let l1_error = try! Y - l1
            //
            //            // multiply how much we missed by the
            //            // slope of the sigmoid at the values in l1
            //            let l1_delta = l1_error * nonlin(x: l1,deriv: true)
            //
            //            // update weights
            //            syn0 = try! syn0 + dot(m1:l0.transpose, m2:l1_delta)
        }
        //
        print("Output After Training:")
        print(l1)
        
        
        
        
        let trainedWeights0 = syn0
        let trainedWeights1 = syn1
        let inputToPredict: Matrix<Float> = [
            [1,0,1, 0, 1, 0 , 1, 0, 1]
        ]
        
        
        let product = try! dot(m1: inputToPredict, m2: syn0)
        let l = nonlin(product)
        let product2 = try! dot(m1: l, m2: syn1)
        let prediction = nonlin(product2)
        
        print("prediction : \(prediction)")
        if prediction[0, 0] > 0.8 {
            print("This is a cross")
        } else {
            print("This is not a cross")
        }
        
        
        
    }
    
    func run4() {
        
        let xs: [Matrix<Float>] = [
            [[1,0,1],
             [0,1,0],
             [1,0,1]] // X
            ,
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
            ,
            [[0,0,1],
             [0,1,0],
             [1,0,1]]
            ,
            [[1,1,1],
             [1,0,1],
             [1,1,1]] // O
            ,
            [[1,1,1],
             [1,1,1],
             [1,1,1]]
        ]
        
        // Transform 2d X in 1d X
        // AKA transofrm
        // [1,0,1],
        // [0,1,0],
        // [1,0,1]]
        // into
        // [1,0,1,0,1,0,1,0,1] // Do we loose spatial info?
        let flattened = xs.map { flatten($0) }
        var X: Matrix<Float> = Matrix(ofShape: (9,9), with: -1)
        for i in 0..<X.numberOfLines {
            for j in 0..<X.numberOfColomns {
                X[i, j] = flattened[i][0,j]
                print(X[i, j])
            }
        }
        
        // output dataset
        let Y: Matrix<Float> = [
            [1, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 1],
            [0, 0],
        ]
        
        // seed random numbers to make calculation
        // deterministic (just a good practice)
        //        np.random.seed(1)
        //
        // initialize weights randomly with mean 0
        //2*np.random.random((3,1)) - 1
        var syn0:Matrix<Float> =  2 * randomMatrix(ofShape: (9,9)) - 1
        var syn1:Matrix<Float> =  2 * randomMatrix(ofShape: (9,2)) - 1
        
        var l1:Matrix<Float>!
        var l2:Matrix<Float>!
        for j in 0...1000 { // try 60 000
            
            
            // Feed forward through layers 0, 1, and 2
            let l0 = X
            let product = try! dot(m1: l0, m2: syn0)
            l1 = nonlin(product)
            
            let product2 = try! dot(m1: l1, m2: syn1)
            l2 = nonlin(product2)
            
            // how much did we miss the target value?
            let l2_error = try! Y - l2
            
            if j % 10000 == 0 {
                print("error")
                //                print("Error:\((np.mean(np.abs(l2_error))))"
            }
            
            
            // in what direction is the target value?
            // were we really sure? if so, don't change too much.
            let l2_delta = l2_error * nonlin(l2, deriv:true)
            
            // how much did each l1 value contribute to the l2 error (according to the weights)?
            let l1_error = try! dot(m1: l2_delta, m2: syn1.transpose) //   l2_delta.dot(syn1.T)
            
            // in what direction is the target l1?
            // were we really sure? if so, don't change too much.
            let l1_delta = l1_error * nonlin(l1,deriv:true)
            
            syn1 = try! syn1 + dot(m1: l1.transpose, m2:l2_delta)  //+= l1.T.dot(l2_delta)
            syn0 = try! syn0 + dot(m1: l0.transpose, m2: l1_delta) //+= l0.T.dot(l1_delta)
        }

        let inputToPredict: Matrix<Float> = [
            [1,1,1, 0, 1, 0 , 1, 0, 1]
        ]
        
        let product = try! dot(m1: inputToPredict, m2: syn0)
        let l = nonlin(product)
        let product2 = try! dot(m1: l, m2: syn1)
        let prediction = nonlin(product2)
        
        print("prediction : \(prediction)")
        if prediction[0, 0] > 0.8 {
            print("This is a cross")
        } else if prediction[0, 1] > 0.8 {
            print("This is an O")
        } else {
            print("This is nothing")
        }
    }
    
    
    
    
    
    func run5() {
        
        let xs: [Matrix<Float>] = [
            [[1,0,0,0,1],
             [0,1,0,1,0],
             [0,0,1,0,0],
             [0,1,0,1,0],
             [1,0,0,0,1]]
            ,
            [[0,0,0,0,1], // no topLeft corner
             [0,1,0,1,0],
             [0,0,1,0,0],
             [0,1,0,1,0],
             [1,0,0,0,1]]
            ,
            [[1,0,0,0,0], // no topRight corner
             [0,1,0,1,0],
             [0,0,1,0,0],
             [0,1,0,1,0],
             [1,0,0,0,1]]
            ,
            [[1,0,0,0,1], // no BottomRight corner
             [0,1,0,1,0],
             [0,0,1,0,0],
             [0,1,0,1,0],
             [1,0,0,0,0]]
            ,
            [[1,0,0,0,1], // no BottomLeft corner
             [0,1,0,1,0],
             [0,0,1,0,0],
             [0,1,0,1,0],
             [0,0,0,0,1]]
            ,
            [[1,0,0,0,0], // only TopLeft corner
            [0,1,0,1,0],
            [0,0,1,0,0],
            [0,1,0,1,0],
            [0,0,0,0,0]]
            ,
            [[0,0,0,0,1], // only TopRight corner
            [0,1,0,1,0],
            [0,0,1,0,0],
            [0,1,0,1,0],
            [0,0,0,0,0]]
            ,
            [[0,0,0,0,0], // only BottomLeft corner
             [0,1,0,1,0],
             [0,0,1,0,0],
             [0,1,0,1,0],
             [0,0,0,0,1]]
            ,
            [[0,0,0,0,0], // only BottomRight corner
            [0,1,0,1,0],
            [0,0,1,0,0],
            [0,1,0,1,0],
            [1,0,0,0,0]]
            ,
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
//            ,
//            ////////
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
            [[0,0,1,1,1],
             [0,0,1,0,0],
             [0,1,0,0,0],
             [0,0,0,0,0],
             [0,0,0,0,1]]
            ,
            [[1,0,1,1,1],
             [0,0,1,0,0],
             [0,1,1,0,0],
             [0,0,0,0,0],
             [0,0,0,0,0]]
            ,
            [[0,0,0,0,1],
             [0,0,0,1,0],
             [0,0,0,0,0],
             [0,1,0,0,0],
             [1,0,0,0,1]]
            ,
            [[1,0,1,1,0],
             [1,0,1,0,1],
             [1,1,0,0,1],
             [1,0,0,0,0],
             [1,1,1,1,1]]
            ,
            [[1,0,1,1,0],
             [1,0,0,0,1],
             [1,0,0,0,1],
             [1,0,0,0,0],
             [1,1,1,1,1]]
        ]
        
        // Transform 2d X in 1d X
        // AKA transofrm
        // [1,0,1],
        // [0,1,0],
        // [1,0,1]]
        // into
        // [1,0,1,0,1,0,1,0,1] // Do we loose spatial info?
        let flattened = xs.map { flatten($0) }
        var X: Matrix<Float> = Matrix(ofShape: (xs.count,25), with: -1)
        for i in 0..<X.numberOfLines {
            for j in 0..<X.numberOfColomns {
                X[i, j] = flattened[i][0,j]
                print(X[i, j])
            }
        }
        
        // output dataset
        let Y: Matrix<Float> = [
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
//            [1],
//            [1],
//            [1],
//            [1],
//            [1],
//            [1],
//            [1],
//            [0],
//            [0],
//            [0],
//            [0],
//            [0],
//            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            ]
        
        var syn0:Matrix<Float> =  2 * randomMatrix(ofShape: (25,9)) - 1
        var syn1:Matrix<Float> =  2 * randomMatrix(ofShape: (9,1)) - 1
        
        var l1:Matrix<Float>!
        var l2:Matrix<Float>!
        for j in 0...100 { // try 60 000
            
            
            // Feed forward through layers 0, 1, and 2
            let l0 = X
            let product = try! dot(m1: l0, m2: syn0)
            l1 = nonlin(product)
            
            let product2 = try! dot(m1: l1, m2: syn1)
            l2 = nonlin(product2)
            
            // how much did we miss the target value?
            let l2_error = try! Y - l2
            
            if j % 10000 == 0 {
                print("error")
                //                print("Error:\((np.mean(np.abs(l2_error))))"
            }
            
            
            // in what direction is the target value?
            // were we really sure? if so, don't change too much.
            let l2_delta = l2_error * nonlin(l2, deriv:true)
            
            // how much did each l1 value contribute to the l2 error (according to the weights)?
            let l1_error = try! dot(m1: l2_delta, m2: syn1.transpose) //   l2_delta.dot(syn1.T)
            
            // in what direction is the target l1?
            // were we really sure? if so, don't change too much.
            let l1_delta = l1_error * nonlin(l1,deriv:true)
            
            syn1 = try! syn1 + dot(m1: l1.transpose, m2:l2_delta)  //+= l1.T.dot(l2_delta)
            syn0 = try! syn0 + dot(m1: l0.transpose, m2: l1_delta) //+= l0.T.dot(l1_delta)
        }
        
        
        
        var inputToPredict: Matrix<Float> =
            [[0,0,0,0,0],
              [0,1,0,1,0],
              [0,0,1,0,0],
              [0,1,0,1,0],
              [0,0,0,0,0]]
        
        inputToPredict = flatten(inputToPredict)
        print(inputToPredict)
        
        let product = try! dot(m1: inputToPredict, m2: syn0)
        let l = nonlin(product)
        let product2 = try! dot(m1: l, m2: syn1)
        let prediction = nonlin(product2)
        
        print("prediction : \(prediction)")
        if prediction[0,0] > 0.8 {
            print("This is a cross")
        } else {
            print("This is nothing")
        }
        
        
        
    }
}

