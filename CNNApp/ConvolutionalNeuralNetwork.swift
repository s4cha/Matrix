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
    
    static let learningRate: Float = 1
}

class ConvolutionalNeuralNetwork {
    
    
    
    var previousGlobalErrorRate:Float = 1
    
    var previousPrediction = [Float]()
    var previousResult:Float = 0
//    var previousErrorRate:Float = 1
//    var Xweights = [Float]()
    var linearVotes = [Float]()
    
    var minError:Float = 1
    
    let convolution = ConvolutionLayer()
    let reLU = RectifiedLinearUnitLayer()
    let pooling = PoolingLayer()
    let fullyConectedLayer = FullyConectedLayer()
    
    func train(with trainingData:  [(Matrix<Float>, [Float])]) {
        
        var previouslyChangedThatWeight = false
        
        for _ in 0...10 { //M
        var isReducingGlobalErrorRate = true
        
        var tunedWeightIndex = 0
        var windex = 0
        
        while isReducingGlobalErrorRate {
            
            
            var errors:Float = 0
            
            for sample in trainingData {
                let matrix = sample.0
//                print("matrix: \(matrix)")
                let minusMatrix = replace(value: 0, by: -1, in: matrix)
                let correctPrediction = sample.1
                
//                print("is X: \(correctPrediction)")
                //            var isReducingErrorRate = true
                
                //            while isReducingErrorRate {
                let prediction = predict(minusMatrix)
                let numberOfPredictions:Float = 2
                
                print(correctPrediction)
                print(prediction)
                
                let errorRate = (abs(correctPrediction[0]-prediction[0])
                    + abs(correctPrediction[1]-prediction[1]))
                    / numberOfPredictions
//                print(errorRate)
                
//                if errorRate < previousErrorRate{
//                    print("Walking down the gradient!")
//                    isReducingErrorRate = true
//                    
//                    
//                    
//                    //try to tune weigths and do it again.
//                    
//                    fullyConectedLayer.weights[0][0] += 0.1
//                    
//                    
//                    
//                } else {
//                    print("Walking up the gradient!")
//                    isReducingErrorRate = false
//                }
//            }
            
            
            
            
                errors += errorRate
            }
            
            let globalErrorRate = errors / Float(trainingData.count)
            
            print("globalErrorRate :\(globalErrorRate)")
            print("previousErrorRate :\(previousGlobalErrorRate)")
            print("weights[0] = :\(fullyConectedLayer.weights[0])")
            print("weights[1] = :\(fullyConectedLayer.weights[1])")
            
            
            
            
            minError = min(previousGlobalErrorRate,minError)
            
            if previousGlobalErrorRate > minError {
                print("wtf")
            }
            
            
    
            if globalErrorRate < previousGlobalErrorRate {
                //                print("Walking down the gradient!")
                
                //try to tune weigths and do it again.
                
                fullyConectedLayer.weights[windex][tunedWeightIndex] += learningRate
                previouslyChangedThatWeight = true
                
                previousGlobalErrorRate = globalErrorRate
            } else {
                //                print("Walking up the gradient!")
                
                if previouslyChangedThatWeight {
                    // set back previous params where it was and go to new param.
                    fullyConectedLayer.weights[windex][tunedWeightIndex] -= learningRate
                    previouslyChangedThatWeight = false
//                    fullyConectedLayer.weights[windex][tunedWeightIndex] += step
                } else {
                    print("noope")
                }
                
                // GO TRY ADJUST NEXT PARAM
                tunedWeightIndex += 1
  
            }
            
            
            
            
            if tunedWeightIndex == 3 && windex == 0 {
                windex = 1
                
                tunedWeightIndex = 0
//                previousGlobalErrorRate = globalErrorRate
            }
            
            if tunedWeightIndex == 3 && windex == 1 {
               isReducingGlobalErrorRate = false
            }
            
            
    
        
            
            
        }
        }
        
        //        for x in xs {
        //            let minusMatrix = replace(value: 0, by: -1, in: x)
        //            let correctAnswer = true
        //            cnn.feed(minusMatrix)
        //            while cnn.predictIsCross() != correctAnswer {
        //                cnn.wrongAnswer(correctAnswer:correctAnswer)
        //            }
        //            print(cnn.Xweights)
        //        }
        //
        
        
        
    }
    
    func predict(_ matrix: Matrix<Float>) -> [Float] {
    
        
        // Convolution
//        let feature1: Matrix<Float> = [
//            [1,-1,-1],
//            [-1,1,-1],
//            [-1,-1,1]
//        ]
        let feature1: Matrix<Float> = [
            [1,-1],
            [-1,1],
        ]
        let c1 = convolution.runOn(matrix: matrix, withFeature: feature1)
        let feature2: Matrix<Float> = [
            [-1,-1,1],
            [-1,1,-1],
            [1,-1,-1]
        ]
//        let c2 = convolution.runOn(matrix: matrix, withFeature: feature2)
//        let feature3: Matrix<Float> = [
//            [1,-1,1],
//            [-1,1,-1],
//            [1,-1,1]
//        ]
//        let feature3: Matrix<Float> = [
//            [-1,-1,-1],
//            [-1,-1,-1],
//            [-1,-1,-1]
//        ]
        //let c3 = convolution.runOn(matrix: matrix, withFeature: feature3)
//        
//        print("Convolution")
//        print(c1)
//        print(c2)
//        print(c3)
        
        // ReLU
        let r1 = reLU.runOn(matrix: c1)
//        let r2 = reLU.runOn(matrix: c2)
//        let r3 = reLU.runOn(matrix: c3)
        
//        print("RectifiedLinearUnitLayer")
//        print(r1)
//        print(r2)
//        print(r3)
        
        // Pooling
        let p1 = pooling.runOn(matrix: r1)
//        let p2 = pooling.runOn(matrix: r2)
//        let p3 = pooling.runOn(matrix: r3)
        
//        print("Pooling")
//        print(p1)
//        print(p2)
//        print(p3)
        
        // 2nd Pooling
        //        let pooling = PoolingLayer()
        //        let p21 = pooling.runOn(matrix: p1)
        //        let p22 = pooling.runOn(matrix: p2)
        //        let p23 = pooling.runOn(matrix: p3)
        //
        //
        //        print(p21)
        //        print(p22)
        //        print(p23)
        
        
        let prediction = fullyConectedLayer.runOn(matrices: [p1])
//        let prediction = fullyConectedLayer.runOn(matrices: [p1, p2, p3])
        
        return prediction
    }
//    
//    func predictIsCross() -> Bool {
//        // X Result :
//        var weightedXVotes = [Float]()
//        for (i,v) in linearVotes.enumerated() {
//            let weight = Xweights[i]
//            weightedXVotes.append(v*weight)
//        }
//        var additionX:Float = 0
//        for v in weightedXVotes {
//            additionX += v
//        }
//        
//        let xResult = additionX / Float(linearVotes.count)
//        
//        previousResult = xResult
//    
//        return xResult > 0.8
//    }
    
//    func wrongAnswer(correctAnswer:Bool) {
//        
//        
//        // calculate error 
//        
//        var correctPrediction:[Float] = correctAnswer ? [1,0] : [1,0]
//        var prediction
//        
//        Xweights[0] = Xweights[0] + 1
//        
//        _ = predictIsCross()
//        
//        //better error rate?
//        
//        
////        // change weight at index 0 and see if error is better.
////       
////        
////
//        var errorRate:Float = 0
//        if correctAnswer {
//            errorRate = 1 - previousResult
//        } else {
//            errorRate = previousResult
//        }
//    
//        print(errorRate)
//        
//        
//        if errorRate < previousErrorRate {
//            //Keep GOing
//            print("Keep GOing")
//        } else {
//            //
//            print("Stop")
//        }
////        var totalErrorRate = 0
////
//        
//        previousErrorRate = errorRate
        
//    }
}


// TODO pick random filters 1 per pass and see wchich ones are the best at clasifying Xs and Os

class FullyConectedLayer {
    
    var linearVotes = [Float]()
    var predictions = [Float]()
    var weights: [[Float]] = [[Float](), [Float]()]
    var weightedVotes: [[Float]] = [[Float](), [Float]()]
    
    func runOn(matrices: [Matrix<Float>]) -> [Float] {
        linearVotes = [Float]()
        for m in matrices {
            for i in 0..<m.backing.count {
                for j in 0..<m.backing.count {
                    linearVotes.append(m[i,j])
                }
            }
        }
        
//        print("linearVotes")
//        print(linearVotes)
        
        
        // Initialize with defaut weights of 1
        
        for i in 0..<2 {
            if weights[i].isEmpty {
                for _ in linearVotes {
                    weights[i].append(1)
                }
            }
        }
        
//        print("weights")
//        print(weights)
        
        
        // Weighted Votes
        for k in 0..<2 {
            for (i,v) in linearVotes.enumerated() {
                let weight = weights[k][i]
                weightedVotes[k].append(v*weight)
            }
        }
        
//        print("weightedVotes")
//        print(weightedVotes)
        
        var predictions = [Float]()
        for i in 0..<2 {
            var addition:Float = 0
            for v in weightedVotes[i] {
                addition += v
            }
            let result = addition / Float(weightedVotes[i].count)
            predictions.append(result)
        }
        return predictions
    }
}



