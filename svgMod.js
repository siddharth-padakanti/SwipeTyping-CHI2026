function getKeyID(key) {
  let numReg = /^\d+$/;
  if (numReg.test(key)) {
    return "Digit" + key;
  }

  let letterReg = /[a-zA-Z]/;
  if (letterReg.test(key) && key.length == 1) {
    return "Key" + key.toUpperCase();
  }

  let keyMapping = {
    ",": "Comma",
    ".": "Period",
    ";": "Semicolon",
    "/": "Slash",
    "-": "Minus",
    "[": "BracketLeft",
    "'": "Quote",
    "=": "Equal",
    "]": "BrackeRight",
    "\\": "Backslash",
    Control: "ControlLeft",
    " ": "Space",
    Backspace: "Backspace",
    Tab: "Tab",
    CapsLock: "CapsLock",
    "`": "Backquote",
    Shift: "ShiftLeft",
    Alt: "AltLeft",
    Fn: "Fn",
  };

  return keyMapping[key];
}

function svgLoad() {
  console.log("LOADED");

  let svgObj = document.getElementById("keyboardOBJ");
  let svgDoc = svgObj.contentDocument;

  svgDoc.addEventListener("mousemove", function (event) {
    myMouseMove(event);
  });
  svgDoc.addEventListener("mouseup", function (event) {
    myMouseUp(event);
  });
  svgDoc.addEventListener("mousedown", function (event) {
    myMouseDown(event);
  });
}

let isMousePressed = false;
let currentKey = null;

function myMouseMove(e) {
  let x = e.clientX;
  let y = e.clientY;
  let coor = "Coordinates: (" + x + "," + y + ")";
  let key = getKeyFromPos(x, y);
  if (isMousePressed) {
    if (key != currentKey) {
      colourKey(key, "gray");
      if (currentKey != null) {
        colourKey(currentKey, "default");
      }
      currentKey = key;
    }
  }
  //document.getElementById("demo").innerHTML = coor;
}

function myMouseDown(e) {
  let x = e.clientX;
  let y = e.clientY;
  let coor = "Down Coordinates: (" + x + "," + y + ")";
  currentKey = getKeyFromPos(x, y);
  if (currentKey != null) {
    isMousePressed = true;
    colourKey(currentKey, "gray");
    setStartPos(x, y, currentKey);
  }
  //document.getElementById("demo").innerHTML = coor;
}

function myMouseUp(e) {
  let x = e.clientX;
  let y = e.clientY;
  let coor = "Up Coordinates: (" + x + "," + y + ")";
  if (isMousePressed) {
    if (currentKey != null) {
      colourKey(currentKey, "default");
    }
    setInput(x, y, currentKey);
    currentKey = null;

    isMousePressed = false;
  }

  //document.getElementById("demo").innerHTML = coor;
}

function colourKey(key, colour) {
  if (key) {
    const obj = document.getElementById("keyboardOBJ");
    const svgDoc = obj.getSVGDocument();

    const keyID = getKeyID(key);
    const keyElem = svgDoc.getElementById(keyID);

    if (keyElem) {
      if (colour == "default") {
        keyElem.setAttribute("fill", "#f7f7f7");
      } else {
        keyElem.setAttribute("fill", colour);
      }
    }
  }
}
