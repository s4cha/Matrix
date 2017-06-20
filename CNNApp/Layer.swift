//
//  Layer.swift
//  CNNApp
//
//  Created by Sacha Durand Saint Omer on 20/06/2017.
//  Copyright © 2017 freshOS. All rights reserved.
//

import Foundation

protocol Layer {
    func runOn(matrix: Matrix<Float>) -> Matrix<Float>
}
