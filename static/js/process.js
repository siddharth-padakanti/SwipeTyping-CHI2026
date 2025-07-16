//global variable
let swipe_length_key = 3;
let swipe_length_threshold = 0.5;
let key_coord = 80; // key distance in (-1, 1) coordinate

let kb_imgWidth = 880;
let kb_imgHeight = 320;
let img_scale = 1;

let keyboardImg;

let isMousePressed = false;
let currentKey = null;

const currentWord = document.getElementById("currentWord");
const display = document.getElementById("display");
const predictionBar = document.getElementById("prediction-bar");

let typedWords = [];
let formattedInput = [];
let startX, startY, startKey;
let entry_result_x = [];
let entry_result_y = [];
let entry_result_gesture = [];
let interp_x = [];
let interp_y = [];

const rows = [
  ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "backspace"],
  ["A", "S", "D", "F", "G", "H", "J", "K", "L", "enter"],
  ["Z", "X", "C", "V", "B", "N", "M", ",", "."],
  [" "]
];

let tap_coords = {
    "Q": [40, 40], "W": [120, 40], "E": [200, 40], "R": [280, 40], "T": [360, 40],
    "Y": [440, 40], "U": [520, 40], "I": [600, 40], "O": [680, 40], "P": [760, 40], "backspace": [840, 40],
    "A": [60, 120], "S": [140, 120], "D": [220, 120], "F": [300, 120], "G": [380, 120],
    "H": [460, 120], "J": [540, 120], "K": [620, 120], "L": [700, 120], "enter": [810, 120],
    "Z": [100, 200], "X": [180, 200], "C": [260, 200], "V": [340, 200], "B": [420, 200],
    "N": [500, 200], "M": [580, 200], ",": [660, 200], ".": [740, 200], " ": [440, 280]
}

function preload() {
  keyboardImg = loadImage(document.getElementById('imgurl').value);
}

function setup() {
  // Remove any old rogue canvas (if reloaded)
  const existing = document.querySelector("canvas");
  if (existing) existing.remove();

  const cnv = createCanvas(800, 400);
  cnv.parent("canvas-container");
  cnv.style("position", "relative"); // <- Make it behave like normal block
  resizeCanvasToFit();

  const imgUrl = document.getElementById("imgurl").value;
  keyboardImg = loadImage(imgUrl, () => {
    console.log("Keyboard image loaded successfully.");
  }, (err) => {
    console.error("Image failed to load:", err);
  });
}


function resizeCanvasToFit() {
  let targetWidth = windowWidth;
  let targetHeight = windowHeight * 0.4; 

  if(targetWidth >= targetHeight / kb_imgHeight * kb_imgWidth){
    // resize based on height
    targetWidth = targetHeight / kb_imgHeight * kb_imgWidth;
  }
  else{
    // resize base on width
    targetHeight = targetWidth / kb_imgWidth * kb_imgHeight;
  }

  resizeCanvas(targetWidth, targetHeight);

  const container = document.getElementById('canvas-container');
  container.style.height = `${targetHeight}px`;
  container.style.width = `${targetWidth}px`;

  img_scale = targetHeight / kb_imgHeight;
}


function windowResized() {
  resizeCanvasToFit();
}

function draw() {
  background(255);

  
  push();
  scale(img_scale);

  image(keyboardImg, 0, 0, kb_imgWidth, kb_imgHeight);
  if (currentKey != null && isMousePressed) {
    colourKey(currentKey);
  }
  drawPath();
  pop();
}


function drawPath(){
  let p_idx = 0;
  for(let i =0;i<entry_result_gesture.length; i++){
    if(entry_result_gesture[i] == "tap"){
      // draw a dot
      fill(247, 220, 111, 100);
      noStroke();
      circle(entry_result_x[p_idx], entry_result_y[p_idx], 25);
      p_idx++;
    }
    else{
      // draw 2 dots
      fill(127, 179, 213, 100);
      noStroke();
      circle(entry_result_x[p_idx], entry_result_y[p_idx], 25);
      circle(entry_result_x[p_idx+1], entry_result_y[p_idx+1], 25);
      stroke(169, 223, 191, 100);
      strokeWeight(5);
      noFill();
      line(entry_result_x[p_idx], entry_result_y[p_idx], entry_result_x[p_idx+1], entry_result_y[p_idx+1]);
      p_idx+=2;
    }
  }

  if(interp_x.length != 0){
    // predicted
    for(let i =0;i<entry_result_x.length-1; i++){
      stroke(33, 47, 61 , 150);
      strokeWeight(5);
      noFill();
      line(entry_result_x[i], entry_result_y[i], entry_result_x[i+1], entry_result_y[i+1]);
    }
    for(let i =0;i<interp_x.length-1; i++){
      fill(146, 43, 33, 100);
      noStroke();
      circle(interp_x[i], interp_y[i], 10);
    }
  }
}

function mousePressed() {
  let x = mouseX / img_scale;
  let y = mouseY / img_scale;
  currentKey = getKeyFromPos(x, y);
  if (currentKey != null) {
    setStartPos(x, y, currentKey);
    isMousePressed = true;
  }
}


function mouseDragged() {
  let x = mouseX / img_scale;
  let y = mouseY / img_scale;
  let key = getKeyFromPos(x, y);
  if (isMousePressed) {
    if (key != currentKey) {
      currentKey = key;
    }
  }
}


function mouseReleased() {
  let x = mouseX / img_scale;
  let y = mouseY / img_scale;
  if (isMousePressed) {
    setInput(x, y, currentKey);
    currentKey = null;

    isMousePressed = false;
  }
}

function colourKey(key) {
  if (key) {
    let kx = tap_coords[key][0];
    let ky = tap_coords[key][1];
    let kw = 80;
    if(key == " "){
      kw = 520;
    }
    else if(key == "enter"){
      kw = 140;
    }
    rectMode(CENTER);
    fill(133, 50);
    noStroke();
    rect(kx, ky, kw, 80);
  }
}

document.addEventListener("keydown", (e) => {
  const key = e.key.toUpperCase();

  if (key === "BACKSPACE") {
    e.preventDefault();
    if (formattedInput.length > 0) {
      const last = formattedInput.pop();

      if (typeof last === "string" && last.includes("degrees")) {
        formattedInput.pop();
        entry_result_x.pop();
        entry_result_x.pop();
        entry_result_y.pop();
        entry_result_y.pop();
        entry_result_gesture.pop();
      } else {
        entry_result_x.pop();
        entry_result_y.pop();
        entry_result_gesture.pop();
      }
    }
    updateDisplay();
  }

  else if (key === "ENTER") {
    e.preventDefault();
    predict();
  }

  else if (key.length === 1 && tap_coords[key]) {
    e.preventDefault();
    const [x, y] = tap_coords[key];

    // Push synthetic tap input
    formattedInput.push(key);
    entry_result_x.push(x);
    entry_result_y.push(y);
    entry_result_gesture.push("tap");
    updateDisplay();
  }
});


/*function keyPressed(event) {
  event.preventDefault();
  colourKey(key, "gray");
}

function keyReleased(event) {
  event.preventDefault();
  colourKey(key, "default");
}*/




function findClosestKey(x, y){
  let min_dist = Number.MAX_SAFE_INTEGER;
  let closest = null;
  for (var key in tap_coords) {
    let kx = tap_coords[key][0];
    let ky = tap_coords[key][1];
    let d = dist(x, y, kx, ky);
    if(d < min_dist){
      min_dist = d;
      closest = key;
    } 
  }
  return closest;
}

function getKeyFromPos(px, py){
  let keyY = Math.floor(map(py, 0, 320, 0, 4));

  if(keyY == 0){
    let keyX = Math.floor(map(px, 0, 880, 0, 11));
    if(keyX < 0 || keyX >= 11){
      return null;
    }
    else{
      return rows[keyY][keyX];
    }
  }
  else if(keyY == 1){
    let keyX = Math.floor(map(px, 20, 740, 0, 9));
    if(keyX < 0 || px >= 880){
      return null;
    }
    else if(keyX >= 9){
      keyX = 9;
      return rows[keyY][keyX];
    }
    else{
      return rows[keyY][keyX];
    }
  }
  else if(keyY == 2){
    let keyX = Math.floor(map(px, 60, 780, 0, 9));
    if(keyX < 0 || keyX >= 9){
      return null;
    }
    else{
      return rows[keyY][keyX];
    }
  }
  else if(keyY == 3){
    let keyX = Math.floor(map(px, 180, 700, 0, 1));
    if(keyX < 0 || keyX >= 1){
      return null;
    }
    else{
      return rows[keyY][keyX];
    }
  }
  else if(keyY == 4){
    return null;
  }
  
  return null;
}

function getAngle(x1, y1, x2, y2){
  let dx = x2 - x1;
  let dy = y2 - y1;
  let angle = degrees(atan2(dy, dx));
  return int((angle + 360) % 360);
}



function getSwipeEnd(x1, y1, x2, y2){
  let dx = x2 - x1;
  let dy = y2 - y1;
  let d_distance = dist(x1, y1, x2, y2);
  let x_res = x1 + dx * swipe_length_key * key_coord / d_distance;
  let y_res = y1 + dy * swipe_length_key * key_coord / d_distance;
  if(x_res < 0){
    x_res = 0;
    let swipe_length_key_update = (x_res - x1) * d_distance / (dx * key_coord);
    y_res = y1 + dy * swipe_length_key_update * key_coord / d_distance;
    if(y_res < 0){
      y_res = 0;
      swipe_length_key_update = (y_res - y1) * d_distance / (dy * key_coord);
      x_res = x1 + dx * swipe_length_key_update * key_coord / d_distance;
    }
    else if(y_res > 240){
      y_res = 240;
      swipe_length_key_update = (y_res - y1) * d_distance / (dy * key_coord);
      x_res = x1 + dx * swipe_length_key_update * key_coord / d_distance;
    }
  }
  else if(x_res > 800){
    x_res = 800;
    let swipe_length_key_update = (x_res - x1) * d_distance / (dx * key_coord);
    y_res = y1 + dy * swipe_length_key_update * key_coord / d_distance;
    if(y_res < 0){
      y_res = 0;
      swipe_length_key_update = (y_res - y1) * d_distance / (dy * key_coord);
      x_res = x1 + dx * swipe_length_key_update * key_coord / d_distance;
    }
    else if(y_res > 240){
      y_res = 240;
      swipe_length_key_update = (y_res - y1) * d_distance / (dy * key_coord);
      x_res = x1 + dx * swipe_length_key_update * key_coord / d_distance;
    }
  }
  else{
    if(y_res < 0){
      y_res = 0;
      swipe_length_key_update = (y_res - y1) * d_distance / (dy * key_coord);
      x_res = x1 + dx * swipe_length_key_update * key_coord / d_distance;
    }
    else if(y_res > 240){
      y_res = 240;
      swipe_length_key_update = (y_res - y1) * d_distance / (dy * key_coord);
      x_res = x1 + dx * swipe_length_key_update * key_coord / d_distance;
    }
  }
  
  return [x_res, y_res];
}

function setStartPos(sx, sy, sk){
  startX = sx;
  startY = sy;
  startKey = sk;
}

function setInput(ex, ey, ek){
  // handle special case first
  if(startKey == "backspace"){
    // press backspace: delete a word or delete current typing word
    if(formattedInput.length > 0){
      // delete current typing word
      clearInput();
    }
    else{
      typedWords.pop();
      const sentence = typedWords.join("");
      display.innerHTML = sentence + '<span class="cursor"></span>';
    }
  }
  else if(startKey == "enter"){
    // press enter: go to next task
  }
  else if(startKey == "," || startKey == "." || startKey == " "){
    // press "," or "." ; predict and add "," or "."
    if(formattedInput.length == 0){
      // no current prediction, add symbol
      typedWords.push(startKey);
      const sentence = typedWords.join("");
      display.innerHTML = sentence + '<span class="cursor"></span>';
      clearInput();
    }
    else{
      predict();
    }    
  }
  else{
    // press all other keys
    if(startKey == ek && dist(startX, startY, ex, ey) < swipe_length_threshold * key_coord){
      // distance smaller than 40 pixels, just a tap
      entry_result_x.push(startX);
      entry_result_y.push(startY);
      formattedInput.push(startKey);
      entry_result_gesture.push("tap");
    }
    else{
      // swipe
      angle = getAngle(startX, startY, ex, ey);
      const [x2_swipe_point, y2_swipe_point] = getSwipeEnd(startX, startY, ex, ey);
      entry_result_x.push(startX);
      entry_result_x.push(x2_swipe_point);
      entry_result_y.push(startY);
      entry_result_y.push(y2_swipe_point);

      formattedInput.push(startKey);
      formattedInput.push(`${angle}degrees`);
      entry_result_gesture.push("swipe");
    }
    updateDisplay();
  }
  
}

function updateDisplay() {
  const text = formattedInput.join(" ");
  currentWord.innerHTML = text + '<span class="cursor"></span>';
}

function clearInput() {
  formattedInput = [];
  entry_result_x = [];
  entry_result_y = [];
  entry_result_gesture = [];
  interp_x = [];
  interp_y = [];
  currentWord.innerHTML = "";
  predictionBar.replaceChildren();
}

function makeArr(startValue, stopValue, cardinality) {
  var arr = [];
  var step = (stopValue - startValue) / (cardinality - 1);
  for (var i = 0; i < cardinality; i++) {
    arr.push(startValue + (step * i));
  }
  return arr;
}

function getTrajectoryChars(entry_x, entry_y){
  let chars_res = [];
  if (entry_x.length == 1){
    // only one point, return same 25 characters
    for (let i=0;i<25;i++){
      chars_res.push(findClosestKey(entry_x[0], entry_y[0]));
    }
  }
  else{
    // get full distance
    let path_dis = 0;
    let original_axis = [];
    original_axis.push(0);
    for(let i=0;i<entry_x.length - 1; i++){
      path_dis += dist(entry_x[i], entry_y[i], entry_x[i + 1], entry_y[i + 1]);
      original_axis.push(path_dis);
    }
    //print(original_axis);
    interp_x = everpolate.linear(makeArr(0, path_dis, 25), original_axis, entry_x);
    interp_y = everpolate.linear(makeArr(0, path_dis, 25), original_axis, entry_y);
    //print(interp_x);
    //print(interp_y);
    for(let i=0;i<interp_x.length; i++){
      chars_res.push(findClosestKey(interp_x[i], interp_y[i]));
    }
  }
    return chars_res;
}

    

function predict() {
  let char_res = getTrajectoryChars(entry_result_x, entry_result_y);
  const input = char_res.join("");
  const count = entry_result_x.length;
  const word = formattedInput.join("").toLowerCase();
  const tapsOnly = entry_result_gesture.every(g => g === "tap");

  
  predictionBar.innerHTML = "Loading...";

  fetch("http://precision.usask.ca/typing/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input, count, word, tapsOnly })
  })
    .then(res => res.json())
    .then(data => {
      if (data.predictions) {
        predictionBar.innerHTML = "";
        data.predictions.forEach(prediction => {
          const box = document.createElement("div");
          box.className = "prediction";
          box.textContent = prediction;
          box.addEventListener("click", () => {
            // Update the display box with selected word
            if(startKey == "," || startKey == "."){
              typedWords.push(prediction + startKey + " ");
            }
            else if(startKey == " "){
              typedWords.push(prediction + startKey);
            }
            const sentence = typedWords.join("");
            display.innerHTML = sentence + '<span class="cursor"></span>';
            // clear all input
            clearInput();
          });
          predictionBar.appendChild(box);
        });
      } else {
        predictionBar.innerHTML = `<p>Error: ${data.error}</p>`;
      }
    })
    .catch(err => {
      predictionBar.innerHTML = `<p>Request failed: ${err}</p>`;
    });
}

// function handleStart(x, y, key, el) {
//   const rect = el.getBoundingClientRect();
//   startX = rect.left + rect.width / 2;
//   startY = rect.top + rect.height / 2;
//   downKey = key;
//   downKeyElement = el;
// }

// function handleEnd(x, y) {
//   if (!downKey || !downKeyElement) return;

//   const rect = downKeyElement.getBoundingClientRect();
//   const insideBox = (
//     x >= rect.left &&
//     x <= rect.right &&
//     y >= rect.top &&
//     y <= rect.bottom
//   );

//   // Always push tap
//   formattedInput.push(downKey);

//   // Only push angle if mouseup/touchend is outside the key box
//   if (!insideBox) {
//     const dx = x - startX;
//     const dy = y - startY;
//     const distance = Math.sqrt(dx * dx + dy * dy);

//     if (distance > 10) {
//       let angle = Math.atan2(dy, dx);
//       angle = (angle * 180 / Math.PI + 360) % 360;
//       angle = Math.round(angle);
//       formattedInput.push(`${angle}degrees`);
//     }
//   }

//   updateDisplay();
//   downKey = null;
//   downKeyElement = null;
// }

// const rows = [
//   ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
//   ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
//   ['Z', 'X', 'C', 'V', 'B', 'N', 'M']
// ];

// rows.forEach(row => {
//   const rowDiv = document.createElement("div");
//   rowDiv.className = "keyboard-row";
//   row.forEach(key => {
//     const btn = document.createElement("div");
//     btn.className = "key";
//     btn.textContent = key;

//     btn.addEventListener("mousedown", e => handleStart(e.clientX, e.clientY, key, btn));
//     btn.addEventListener("touchstart", e => {
//       const touch = e.touches[0];
//       handleStart(touch.clientX, touch.clientY, key, btn);
//       e.preventDefault();
//     });

//     rowDiv.appendChild(btn);
//   });
//   keyboard.appendChild(rowDiv);
// });

// document.addEventListener("mouseup", e => handleEnd(e.clientX, e.clientY));
// document.addEventListener("touchend", e => {
//   const touch = e.changedTouches[0];
//   handleEnd(touch.clientX, touch.clientY);
// });

// document.addEventListener("keydown", (e) => {
//   const key = e.key;
//   if (key === "Backspace") {
//     e.preventDefault();
//     formattedInput.pop();
//     updateDisplay();
//   } else if (key === "Enter") {
//     e.preventDefault();
//     predict();
//   } else if (key.length === 1 && /^[a-zA-Z0-9]$/.test(key)) {
//     formattedInput.push(key.toUpperCase());
//     updateDisplay();
//   }
// });