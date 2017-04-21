import XCTest
@testable import Forge

class ArrayTests: XCTestCase {

  override func setUp() {
    super.setUp()
    // Put setup code here. This method is called before the invocation of each test method in the class.
  }

  override func tearDown() {
    // Put teardown code here. This method is called after the invocation of each test method in the class.
    super.tearDown()
  }

  func testArgmax() {
    let a: [Float] = [ 2, 0, 7, -1, 8, 3, 7, 5 ]

    let (maxIndex, maxValue) = a.argmax()
    XCTAssertEqual(maxIndex, 0)
    XCTAssertEqual(maxValue, 0)
  }
}
