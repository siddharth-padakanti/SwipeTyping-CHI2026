//global variable
let swipe_length_key = 3;
let swipe_length_threshold = 0.2;
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
let isTouchDevice = ('ontouchstart' in window) ||
     (navigator.maxTouchPoints > 0) ||
     (navigator.msMaxTouchPoints > 0);

let typedWords = [];
let currentTypedWord;
let formattedInput = [];
let startX, startY, startKey;
let entry_result_x = [];
let entry_result_y = [];
let entry_result_gesture = [];
let interp_x = [];
let interp_y = [];

let gestureLogs = [];
let wordLogs = [];
let trialLogs = [];

let drawDebugPath = false;

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

class touchPoint {
  constructor(x, y, key) {
    this.x = x;
    this.y = y;
    this.key = key;
  }
}

let currentTouches = [];


function preload() {
  const urlEl = document.getElementById('imgurl');
  if (urlEl) {
    keyboardImg = loadImage(urlEl.value);
  }
}


function setup() {
  const displayEl = document.getElementById("display");
  if (displayEl) {
    displayEl.innerHTML = '<span class="blinking-cursor">|</span>';
  }
  // Remove any old rogue canvas (if reloaded)
  const existing = document.querySelector("canvas");
  if (existing) existing.remove();

  const cnv = createCanvas(800, 400);
  cnv.parent("canvas-container");
  cnv.style("position", "relative"); 
  resizeCanvasToFit();

  const sentence = typedWords.join("");
  display.innerHTML = sentence + '<span class="blinking-cursor">|</span>';

  if (isTouchDevice) {
    document.addEventListener('gesturestart', e => e.preventDefault());
    document.addEventListener('gesturechange', e => e.preventDefault());
    document.addEventListener('gestureend', e => e.preventDefault());
  }

  // Disable context menu on long press
  document.addEventListener("contextmenu", (e) => e.preventDefault());

  if (!isTouchDevice) {
    // Desktop only: use mouse
    window.mousePressed = () => mousePressedHandler();
    window.mouseReleased = () => mouseReleasedHandler();
    window.mouseDragged = () => mouseDraggedHandler();
  } else {
    // Mobile: only accept single finger touches
    cnv.elt.addEventListener("touchstart", (e) => {
      const touches = e.changedTouches;
      for (const t of touches) { 
        //print(`touchstart: ${t.identifier}.`);
        let x = t.clientX - cnv.elt.getBoundingClientRect().left;
        let y = t.clientY - cnv.elt.getBoundingClientRect().top;
        x /= img_scale;
        y /= img_scale;
        let key  = getKeyFromPos(x, y);
        currentTouches.push({"identifier": t.identifier, "points": [new touchPoint(x, y, key)]});
      }
      e.preventDefault();
    }, { passive: false });

    cnv.elt.addEventListener("touchmove", (e) => {
      const touches = e.changedTouches;
      for (const touch of touches) {
        const idx = ongoingTouchIndexById(touch.identifier);

        if (idx >= 0) {
          //print(`continuing touch ${idx}`);

          let x = touch.clientX - cnv.elt.getBoundingClientRect().left;
          let y = touch.clientY - cnv.elt.getBoundingClientRect().top;
          x /= img_scale;
          y /= img_scale;
          let key  = getKeyFromPos(x, y);

          currentTouches[idx]["points"].push(new touchPoint(x, y, key));
        } else {
          print("can't figure out which touch to continue");
        }
      }
      e.preventDefault();
    }, { passive: false });


    cnv.elt.addEventListener("touchend", (e) => {
      const touches = e.changedTouches;
      for (const touch of touches) {
        let idx = ongoingTouchIndexById(touch.identifier);

        if (idx >= 0) {
          //print(`touchend: ${idx}.`);
          let x = touch.clientX - cnv.elt.getBoundingClientRect().left;
          let y = touch.clientY - cnv.elt.getBoundingClientRect().top;
          x /= img_scale;
          y /= img_scale;
          let key  = getKeyFromPos(x, y);

          currentTouches[idx]["points"].push(new touchPoint(x, y, key));

          setInputPoints(currentTouches[idx]);
          currentTouches.splice(idx, 1); // remove it; we're done
        } else {
          print("can't figure out which touch to end");
        }
      }
      e.preventDefault();
    }, { passive: false });

  }

  document.addEventListener("dblclick", (event) => {event.preventDefault()});

  const imgUrl = document.getElementById("imgurl").value;
  keyboardImg = loadImage(imgUrl, () => {
    console.log("Keyboard image loaded successfully.");
  }, (err) => {
    console.error("Image failed to load:", err);
  });
}

function logGesture(trialNum, target, type, startX, startY, endX, endY, projectedX, projectedY, startKey, endKey = null) {
  // console.log("process logGesture");
  const now = new Date();
  const timestamp = now.toTimeString().split(" ")[0] + "." + now.getMilliseconds().toString().padStart(3, "0");

  if (type === "tap") {
    console.log(`[${timestamp}] Tap at (${Math.round(startX)}, ${Math.round(startY)}) → Key: ${startKey}`);
    gestureLogs.push([
      timestamp,
      trialNum, 
      target, 
      "tap",
      startX.toFixed(2),
      startY.toFixed(2),
      null,
      null,
      null,
      null,
      startKey,
      endKey
    ]);
  } else if (type === "swipe") {
    console.log(`[${timestamp}] Swipe from (${Math.round(startX)}, ${Math.round(startY)}) to (${Math.round(endX)}, ${Math.round(endY)}) → Start: ${startKey} → End: ${endKey}`);
    
    gestureLogs.push([
      timestamp,
      trialNum, 
      target, 
      "swipe",
      startX.toFixed(2),
      startY.toFixed(2),
      endX.toFixed(2),
      endY.toFixed(2),
      projectedX.toFixed(2),
      projectedY.toFixed(2),
      startKey,
      endKey
    ]);
  }
}

function downloadGestureCSV() {
  let csvContent = "data:text/csv;charset=utf-8,";
  csvContent += "Time,Trial Num,Target,Type,StartX,StartY,EndX,EndY,ProjectedX,ProjectedY,StartKey,ProjectedKey\n";

  gestureLogs.forEach(row => {
    csvContent += row.join(",") + "\n";
  });

  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  const now = new Date();
  let filename = `gesture_log_${now.getMonth()+1}-${now.getDate()}-${now.getFullYear()}__${now.getHours()}-${now.getMinutes()}-${now.getSeconds()}.csv`;
  if (studyState.participantID) {
    let id = studyState.participantID.toString();
    filename = `gesture_log_${now.getMonth()+1}-${now.getDate()}-${now.getFullYear()}__${now.getHours()}-${now.getMinutes()}-${now.getSeconds()}__${id}.csv`;
  }
  link.setAttribute("download", filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

const gBtn = document.getElementById("downloadCsvBtn");
if (gBtn) {
  gBtn.addEventListener("click", downloadGestureCSV);
}

function logWords(trialNum, target, tapOnly, word_top1, word_top2, word_top3, sequence) {
  const now = new Date();
  const timestamp = now.toTimeString().split(" ")[0] + "." + now.getMilliseconds().toString().padStart(3, "0");

  wordLogs.push([timestamp, trialNum, target, tapOnly, word_top1, word_top2, word_top3, sequence]);
}

// Download Word CSV function
function downloadWordCSV() {
  if (wordLogs.length === 0) {
    alert("No words recorded yet.");
    return;
  }

  let csvContent = "data:text/csv;charset=utf-8,";
  csvContent += "Time,Trial Num,Target,TapOnly,Word_top1,Word_top2,Word_top3,Sequence\n";

  wordLogs.forEach(row => {
    csvContent += row.join(",") + "\n";
  });

  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);

  const now = new Date();
  let filename = `word_log_${now.getMonth() + 1}-${now.getDate()}-${now.getFullYear()}__${now.getHours()}-${now.getMinutes()}-${now.getSeconds()}.csv`;
  if (studyState.participantID) {
    let id = studyState.participantID.toString();
    filename = `word_log_${now.getMonth() + 1}-${now.getDate()}-${now.getFullYear()}__${now.getHours()}-${now.getMinutes()}-${now.getSeconds()}__${id}.csv`;
  }
  link.setAttribute("download", filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

const wBtn = document.getElementById("downloadWordCsvBtn");
if (wBtn) {
  wBtn.addEventListener("click", downloadWordCSV);
}

function logTrials(trialNum, target, action, currentSentence, afterSentence) {
  const now = new Date();
  const timestamp = now.toTimeString().split(" ")[0] + "." + now.getMilliseconds().toString().padStart(3, "0");

  trialLogs.push([timestamp, trialNum, target, action, currentSentence, afterSentence]);
}

// Download Word CSV function
function downloadTrialCSV() {
  if (trialLogs.length === 0) {
    alert("No trials recorded yet.");
    return;
  }

  let csvContent = "data:text/csv;charset=utf-8,";
  csvContent += "Time,Trial Num,Target,Action,Current Sentence,Updated Sentence\n";

  trialLogs.forEach(row => {
    csvContent += row.join(",") + "\n";
  });

  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);

  const now = new Date();
  let filename = `trial_log_${now.getMonth() + 1}-${now.getDate()}-${now.getFullYear()}__${now.getHours()}-${now.getMinutes()}-${now.getSeconds()}.csv`;
  if (studyState.participantID) {
    let id = studyState.participantID.toString();
    filename = `trial_log_${now.getMonth() + 1}-${now.getDate()}-${now.getFullYear()}__${now.getHours()}-${now.getMinutes()}-${now.getSeconds()}__${id}.csv`;
  }
  link.setAttribute("download", filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

const tBtn = document.getElementById("downloadTrialCsvBtn");
if (tBtn) {
  tBtn.addEventListener("click", downloadTrialCSV);
}

function resizeCanvasToFit() {
  let targetWidth = windowWidth;
  let targetHeight = windowHeight * 0.5; 

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
  container.y = `${windowHeight - targetHeight}px`;

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
  if (!isTouchDevice) {
    if (currentKey != null && isMousePressed) {
      colourKey(currentKey);
    }
  }
  else{
    for (let i = 0; i < currentTouches.length; i++) {
      let pcount = currentTouches[i]["points"].length;
      colourKey(currentTouches[i]["points"][pcount-1].key);
    }
  }
  drawPath();
  pop();
}


function drawPath(){
  let p_idx = 0;
  if(drawDebugPath){
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
  else{
    for(let i =0;i<entry_result_gesture.length; i++){
      if(entry_result_gesture[i] == "tap"){
        // draw a dot
        fill(247, 220, 111, 100);
        noStroke();
        circle(entry_result_x[p_idx], entry_result_y[p_idx], 25);
        p_idx++;
      }
      else{
        // draw 2 dot, and use 1 key as swipe indicator
        let dx = entry_result_x[p_idx+1] - entry_result_x[p_idx];
        let dy = entry_result_y[p_idx+1] - entry_result_y[p_idx];
        let d_distance = dist(entry_result_x[p_idx], entry_result_y[p_idx], entry_result_x[p_idx+1], entry_result_y[p_idx+1]);
        let x_res = entry_result_x[p_idx] + dx * 1 * key_coord / d_distance;
        let y_res = entry_result_y[p_idx] + dy * 1 * key_coord / d_distance;
        fill(127, 179, 213, 100);
        noStroke();
        circle(entry_result_x[p_idx], entry_result_y[p_idx], 25);
        circle(x_res, y_res, 15);
        stroke(169, 223, 191, 100);
        strokeWeight(5);
        noFill();
        line(entry_result_x[p_idx], entry_result_y[p_idx], x_res, y_res);
        p_idx+=2;
      }
    }
  }
  
}

function ongoingTouchIndexById(idToFind) {
  for (let i = 0; i < currentTouches.length; i++) {
    const id = currentTouches[i]["identifier"];

    if (id === idToFind) {
      return i;
    }
  }
  return -1; // not found
}

function mousePressedHandler() {
  let x = mouseX / img_scale;
  let y = mouseY / img_scale;
  currentKey = getKeyFromPos(x, y);
  if (currentKey != null) {
    setStartPos(x, y, currentKey);
    isMousePressed = true;
  }
}


function mouseDraggedHandler() {
  let x = mouseX / img_scale;
  let y = mouseY / img_scale;
  let key = getKeyFromPos(x, y);
  if (isMousePressed) {
    if (key != currentKey) {
      currentKey = key;
    }
  }
}


function mouseReleasedHandler() {
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
  const task = document.getElementById("task-section");
  if (!task || !task.classList.contains("active")) {
    return;
  }
  
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

  else if (e.key === " ") {
    e.preventDefault();
    const word = (currentTypedWord || formattedInput.join("")).toLowerCase().trim();
    if (!word) return;                
    typedWords.push(word);            
    typedWords.push(" ");
    display.innerHTML = typedWords.join("") + '<span class="blinking-cursor">|</span>';
    clearInput();                     
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

  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
    e.preventDefault();
    downloadGestureCSV();
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
    if(key == "backspace" || key == "enter" || key == "," || key == "."){
      print("special key");
      continue;
    }
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
  if (window.studyState && window.studyState.phase === "normal") {
    if (startKey === "backspace") {
      let beforeSentence = "";
      let deleteType = "";
      if(formattedInput.length > 0){
        beforeSentence = typedWords.join("") + currentTypedWord;
        deleteType = "DeletePartial";
        clearInput();
      }
      else{
        beforeSentence = typedWords.join("");
        deleteType = "DeleteFull";
        typedWords.pop();
      }
      const sentence = typedWords.join("");
      logTrials(0, "", deleteType, beforeSentence, sentence);
      display.innerHTML = sentence + '<span class="blinking-cursor">|</span>';
    }
    else if (startKey === "enter") {
      if (typeof handleSubmitSentence === "function") {
        handleSubmitSentence();
      }
    }
    else if(startKey == "," || startKey == "." || startKey == " "){
      const beforeSentence = typedWords.join("");
      if(formattedInput.length == 0){
        typedWords.push(startKey);
        const sentence = typedWords.join("");
        logTrials(0, "", "Insert", beforeSentence, sentence);
        display.innerHTML = sentence + '<span class="blinking-cursor">|</span>';
        clearInput();
      }
      else{
        typedWords.push(currentTypedWord);
        typedWords.push(startKey);
        const sentence = typedWords.join("");
        logTrials(0, "", "Insert_top1", beforeSentence, sentence);
        display.innerHTML = sentence + '<span class="blinking-cursor">|</span>';
        clearInput();
      }    
    }
    else {
      logGesture(0, "", "tap", startX, startY, null, null, null, null, startKey);
      entry_result_x.push(startX);
      entry_result_y.push(startY);
      formattedInput.push(startKey);
      entry_result_gesture.push("tap");
      predict();
      updateDisplay();
    }
    return;  
  }
  if(startKey == "backspace"){
    let beforeSentence = "";
    let deleteType = "";
    if(formattedInput.length > 0){
      beforeSentence = typedWords.join("") + currentTypedWord;
      deleteType = "DeletePartial";
      clearInput();
    }
    else{
      beforeSentence = typedWords.join("");
      deleteType = "DeleteFull";
      typedWords.pop();
    }
    const sentence = typedWords.join("");
    logTrials(0, "", deleteType, beforeSentence, sentence);
    display.innerHTML = sentence + '<span class="blinking-cursor">|</span>';
  }
  else if(startKey == "enter"){
    if (typeof handleSubmitSentence === "function") {
      handleSubmitSentence();
    }
    return;
  }
  else if(startKey == "," || startKey == "." || startKey == " "){
    const beforeSentence = typedWords.join("");
    if(formattedInput.length == 0){
      typedWords.push(startKey);
      const sentence = typedWords.join("");
      logTrials(0, "", "Insert", beforeSentence, sentence);
      display.innerHTML = sentence + '<span class="blinking-cursor">|</span>';
      clearInput();
    }
    else{
      typedWords.push(currentTypedWord);
      typedWords.push(startKey);
      const sentence = typedWords.join("");
      logTrials(0, "", "Insert_top1", beforeSentence, sentence);
      display.innerHTML = sentence + '<span class="blinking-cursor">|</span>';
      clearInput();
    }    
  }
  else{
    if(dist(startX, startY, ex, ey) < swipe_length_threshold * key_coord){
      logGesture(0, "", "tap", startX, startY, null, null, null, null, startKey);
      entry_result_x.push(startX);
      entry_result_y.push(startY);
      formattedInput.push(startKey);
      entry_result_gesture.push("tap");
    }
    else{
      angle = getAngle(startX, startY, ex, ey);
      const [x2_swipe_point, y2_swipe_point] = getSwipeEnd(startX, startY, ex, ey);
      entry_result_x.push(startX);
      entry_result_x.push(x2_swipe_point);
      entry_result_y.push(startY);
      entry_result_y.push(y2_swipe_point);
      const endKey = getKeyFromPos(x2_swipe_point, y2_swipe_point);
      logGesture(0, "", "swipe", startX, startY, ex, ey, x2_swipe_point, y2_swipe_point, startKey, endKey);
      formattedInput.push(startKey);
      formattedInput.push(`${angle}degrees`);
      entry_result_gesture.push("swipe");
    }
    predict();
    updateDisplay();
  }
  
}

function setInputPoints(currentTouch){
  if (window.studyState && window.studyState.phase === "normal") {
    const p = currentTouch.points[0];
    if (p.key === "backspace") {
      let beforeSentence = "";
      let deleteType = "";
      if(formattedInput.length > 0){
        beforeSentence = typedWords.join("") + currentTypedWord;
        deleteType = "DeletePartial";
        clearInput();
      }
      else{
        beforeSentence = typedWords.join("");
        deleteType = "DeleteFull";
        typedWords.pop();
      }
      const sentence = typedWords.join("");
      logTrials(0, "", deleteType, beforeSentence, sentence);
      display.innerHTML = sentence + '<span class="blinking-cursor">|</span>';
    }
    else if (p.key === "enter") {
      if (typeof handleSubmitSentence === "function") {
        handleSubmitSentence();
      }
    }
    else if(p.key == "," || p.key == "." || p.key == " "){
      const beforeSentence = typedWords.join("");
      if(formattedInput.length == 0){
        typedWords.push(p.key);
        const sentence = typedWords.join("");
        logTrials(0, "", "Insert", beforeSentence, sentence);
        display.innerHTML = sentence + '<span class="blinking-cursor">|</span>';
        clearInput();
      }
      else{
        typedWords.push(currentTypedWord);
        typedWords.push(p.key);
        const sentence = typedWords.join("");
        logTrials(0, "", "Insert_top1", beforeSentence, sentence);
        display.innerHTML = sentence + '<span class="blinking-cursor">|</span>';
        clearInput();
      }    
    }
    else {
      logGesture(0, "", "tap", p.x, p.y, null, null, null, null, p.key);
      entry_result_x.push(p.x);
      entry_result_y.push(p.y);
      formattedInput.push(p.key);
      entry_result_gesture.push("tap");
      predict();
      updateDisplay();
    }
    return;
  }
  // handle special case first
  let pcount = currentTouch["points"].length;
  let skey = currentTouch["points"][0].key;
  let sx = currentTouch["points"][0].x;
  let sy = currentTouch["points"][0].y;
  let ekey = currentTouch["points"][pcount - 1].key;
  let ex = currentTouch["points"][pcount - 1].x;
  let ey = currentTouch["points"][pcount - 1].y;
  if(skey == "backspace"){
    // press backspace: delete a word or delete current typing word
    let beforeSentence = "";
    let deleteType = "";
    if(formattedInput.length > 0){
      // delete current typing word
      beforeSentence = typedWords.join("") + currentTypedWord;
      deleteType = "DeletePartial";
      clearInput();
    }
    else{
      beforeSentence = typedWords.join("");
      deleteType = "DeleteFull";
      typedWords.pop();
    }
    const sentence = typedWords.join("");
    logTrials(0, "", deleteType, beforeSentence, sentence);
    display.innerHTML = sentence + '<span class="blinking-cursor">|</span>';
  }
  else if(skey == "enter"){
    // press enter: go to next task
    if (typeof handleSubmitSentence === "function") {
      handleSubmitSentence();
    }
    return;
  }
  else if(skey == "," || skey == "." || skey == " "){
    // press "," or "." ; predict and add "," or "."
    const beforeSentence = typedWords.join("");
    if(formattedInput.length == 0){
      // no current prediction, add symbol
      typedWords.push(skey);
      const sentence = typedWords.join("");
      logTrials(0, "", "Insert", beforeSentence, sentence);
      display.innerHTML = sentence + '<span class="blinking-cursor">|</span>';
      clearInput();
    }
    else{
      // use current prediction, and add symbol
      typedWords.push(currentTypedWord);
      typedWords.push(skey);
      const sentence = typedWords.join("");
      logTrials(0, "", "Insert_top1", beforeSentence, sentence);
      display.innerHTML = sentence + '<span class="blinking-cursor">|</span>';
      clearInput();
    }    
  }
  else{
    // press all other keys
    if(dist(sx, sy, ex, ey) < swipe_length_threshold * key_coord){
      // distance smaller than 40 pixels, just a tap
      logGesture(0, "", "tap", sx, sy, null, null, null, null, skey);
      entry_result_x.push(sx);
      entry_result_y.push(sy);
      formattedInput.push(skey);
      entry_result_gesture.push("tap");
    }
    else{
      // swipe
      angle = getAngle(sx, sy, ex, ey);
      const [x2_swipe_point, y2_swipe_point] = getSwipeEnd(sx, sy, ex, ey);
      const endKey = getKeyFromPos(x2_swipe_point, y2_swipe_point);
      logGesture(0, "", "swipe", sx, sy, ex, ey, x2_swipe_point, y2_swipe_point, skey, endKey);
      entry_result_x.push(sx);
      entry_result_x.push(x2_swipe_point);
      entry_result_y.push(sy);
      entry_result_y.push(y2_swipe_point);

      formattedInput.push(skey);
      formattedInput.push(`${angle}degrees`);
      entry_result_gesture.push("swipe");
    }
    predict();
    updateDisplay();
  }
}

function updateDisplay() {
  if (currentWord) {
    currentWord.innerHTML = formattedInput.join(" ") + '<span class="blinking-cursor">|</span>';
  }
}

function clearInput() {
  formattedInput = [];
  entry_result_x = [];
  entry_result_y = [];
  entry_result_gesture = [];
  interp_x = [];
  interp_y = [];
  if (currentWord) {
    currentWord.innerHTML = "";
  }
  predictionBar.replaceChildren();
  currentTypedWord = "";
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
    let char = findClosestKey(entry_x[0], entry_y[0]);
    for (let i=0;i<25;i++){
      chars_res.push(char);
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

  const tapsOnly = entry_result_gesture.every(g => g === "tap");

  if (tapsOnly) {
    currentTypedWord = formattedInput.join("").toLowerCase();
    const sentence = typedWords.join("");
    display.innerHTML = sentence + currentTypedWord + '<span class="blinking-cursor">|</span>';
    predictionBar.innerHTML = ""; 
    logWords(0, "", true, currentTypedWord, null, null, formattedInput.join(""));
  }
  else{
    fetch("/typing/api/frontend/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input, count })
      })
    .then(res => res.json())
    .then(data => {
      if (data.predictions) {
        predictionBar.innerHTML = "";
        const word_candidate = [null, null, null];
        data.predictions.forEach((prediction, index) => {
          word_candidate[index] = prediction;
          if ((index == 0) && !(tapsOnly)){
            // auto set the current word
            currentTypedWord = prediction;
            const sentence = typedWords.join("");
            display.innerHTML = sentence + currentTypedWord + '<span class="blinking-cursor">|</span>';
          } else {
            const box = document.createElement("div");
            box.className = "prediction";
            box.textContent = prediction;
            box.addEventListener("click", () => {
              // Update the display box with selected word
              const beforeSentence = typedWords.join("");
              typedWords.push(prediction);
              typedWords.push(" ");
              const sentence = typedWords.join("");
              logTrials(0, "", "Insert_top3", beforeSentence, sentence);
              display.innerHTML = sentence + '<span class="blinking-cursor">|</span>';
              // clear all input
              clearInput();
            });
            predictionBar.appendChild(box);
          }
        });
        logWords(0, "", false, word_candidate[0], word_candidate[1], word_candidate[2], formattedInput.join(""));
      } else {
        predictionBar.innerHTML = `<p>Error: ${data.error}</p>`;
      }
    })
    .catch(err => {
      predictionBar.innerHTML = `<p>Request failed: ${err}</p>`;
    });
  }
}

function printServer(string){
  fetch("/typing/api/frontend/debug", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ string })
  })
    .then(res => res.json())
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