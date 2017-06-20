//
//  Matrix.swift
//  CNNApp
//
//  Created by Sacha Durand Saint Omer on 19/06/2017.
//  Copyright © 2017 freshOS. All rights reserved.
//

import Foundation


struct Matrix<T> {
    var backing:[[T]]
}

extension Matrix: ExpressibleByArrayLiteral {
    
    init(arrayLiteral elements: [T]...) {
        if let count = elements.first?.count {
            for row in elements {
                if row.count != count {
                    fatalError("Matrix must have same number of rows and columns")
                }
            }
        }
        self.init(backing: elements)
    }
    
    subscript(row: Int) -> [T] {
        get {
            return backing[row]
        }
        set {
            backing[row] = newValue
        }
    }
    
    var numberOfLines:Int {
        return backing.count
    }
}

extension Matrix: CustomStringConvertible {
    
    var description: String {
        var s = ""
        for lines in backing {
            s += lines.description
            s += "\n"
        }
        return s
    }
}
