//global variable
let swipe_length_key = 3;
let swipe_length_threshold = 0.5;
let key_coord = 80; // key distance in (-1, 1) coordinate

let keyboardImg;

let isMousePressed = false;
let currentKey = null;

const swipeInput = document.getElementById("swipeInput");
const display = document.getElementById("display");

let formattedInput = [];
let startX, startY, startKey;
let entry_result_x = [];
let entry_result_y = [];
let entry_result_gesture = [];
let interp_x = [];
let interp_y = [];

const rows = [
  ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
  ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
  ["Z", "X", "C", "V", "B", "N", "M"],
  [" "]
];

let tap_coords = {
    "Q": [40, 40], "W": [120, 40], "E": [200, 40], "R": [280, 40], "T": [360, 40],
    "Y": [440, 40], "U": [520, 40], "I": [600, 40], "O": [680, 40], "P": [760, 40],
    "A": [60, 120], "S": [140, 120], "D": [220, 120], "F": [300, 120], "G": [380, 120],
    "H": [460, 120], "J": [540, 120], "K": [620, 120], "L": [700, 120],
    "Z": [100, 200], "X": [180, 200], "C": [260, 200], "V": [340, 200], "B": [420, 200],
    "N": [500, 200], "M": [580, 200], " ": [390, 280]
}

function preload() {
  keyboardImg = loadImage(document.getElementById('imgurl').value);
}

function setup() {
  createCanvas(800, 400);

}

function draw() {
  background(255);
  image(keyboardImg, 0, 0, 800, 320);
  if (currentKey != null && isMousePressed) {
    colourKey(currentKey);
  }
  drawPath();
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
  let x = mouseX;
  let y = mouseY;
  let coor = "Down Coordinates: (" + x + "," + y + ")";
  currentKey = getKeyFromPos(x, y);
  if (currentKey != null) {
    isMousePressed = true;
    setStartPos(x, y, currentKey);
  }
}

function mouseDragged() {
  let x = mouseX;
  let y = mouseY;
  let coor = "Coordinates: (" + x + "," + y + ")";
  let key = getKeyFromPos(x, y);
  if (isMousePressed) {
    if (key != currentKey) {
      currentKey = key;
    }
  }
}

function mouseReleased() {
  let x = mouseX;
  let y = mouseY;
  let coor = "Up Coordinates: (" + x + "," + y + ")";
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
      kw = 440;
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
    let keyX = Math.floor(map(px, 0, 800, 0, 10));
    if(keyX < 0 || keyX >= 10){
      return null;
    }
    else{
      return rows[keyY][keyX];
    }
  }
  else if(keyY == 1){
    let keyX = Math.floor(map(px, 20, 740, 0, 9));
    if(keyX < 0 || keyX >= 9){
      return null;
    }
    else{
      return rows[keyY][keyX];
    }
  }
  else if(keyY == 2){
    let keyX = Math.floor(map(px, 60, 620, 0, 7));
    if(keyX < 0 || keyX >= 7){
      return null;
    }
    else{
      return rows[keyY][keyX];
    }
  }
  else if(keyY == 3){
    let keyX = Math.floor(map(px, 180, 620, 0, 1));
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

function updateDisplay() {
  const text = formattedInput.join(" ");
  display.innerHTML = text + '<span class="cursor"></span>';
  swipeInput.value = text;
}

function clearInput() {
  formattedInput = [];
  entry_result_x = [];
  entry_result_y = [];
  entry_result_gesture = [];
  interp_x = [];
  interp_y = [];
  swipeInput.value = "";
  display.innerHTML = '<span class="cursor"></span>';
  document.getElementById("results").innerHTML = "";
  document.getElementById("demo").innerHTML = "";
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
  //print(entry_result_x);
  //print(entry_result_y);

  if (entry_result_x.length === 0 || entry_result_y.length === 0) {
    document.getElementById("demo").innerHTML = "Empty Input";
    document.getElementById("results").innerHTML = "";
  }

  let char_res = getTrajectoryChars(entry_result_x, entry_result_y);
  document.getElementById("demo").innerHTML = char_res.join("");

  const input = char_res.join("");
  const count = entry_result_x.length;
  const word = formattedInput.join("").toLowerCase();
  const tapsOnly = entry_result_gesture.every(g => g === "tap");
  const results = document.getElementById("results");
  results.innerHTML = "Loading...";

  fetch("http://localhost:1111/typing/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input, count, word, tapsOnly })
  })
    .then(res => res.json())
    .then(data => {
      if (data.predictions) {
        results.innerHTML = `<h3>Predictions:</h3><ul>${data.predictions.map(word => `<li>${word}</li>`).join('')}</ul>`;
      } else {
        results.innerHTML = `<p>Error: ${data.error}</p>`;
      }
    })
    .catch(err => {
      results.innerHTML = `<p>Request failed: ${err}</p>`;
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