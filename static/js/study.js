// study.js

// ==== Monkey‐patch logging so we only record in main study, but always track swipes ====
window.STUDY_LOGGING_ENABLED = false;
const _origLogGesture = window.logGesture;
const _origLogWords   = window.logWords;

window.logGesture = function(type, sx, sy, ex, ey, skey, ekey) {
  // always count swipes for practice‐check
  if (type === "swipe") studyState.swipeCount++;
  // only record full logs in main study
  if (STUDY_LOGGING_ENABLED) {
    _origLogGesture(type, sx, sy, ex, ey, skey, ekey);
  }
};

window.logWords = function() {
  if (STUDY_LOGGING_ENABLED) {
    _origLogWords();
  }
};


// ==== Study state and sentence lists ====
const studyState = {
  participantID: null,
  phase: null,          // "normal", "swipe", "main"
  trialIndex: 0,
  sentences: [],
  swipeCount: 0
};
window.studyState = studyState;

// fill these with your actual trial sentences:
const NORMAL_SENTENCES = [
  "i love this bright sunny day",
  "we enjoy a calm evening walk",
  "the calm water reflects the sky",
  "she smiles with joy and grace",
  "music brings a feeling of peace",
  "our family gather at the table",
  "he reads a good book daily",
  "walk with courage and stay strong",
  "dream big and never give up",
  "help others and share your time",
  "kind words can heal the heart",
  "art and music make our lives better",
  "nature provides beauty and comfort",
  "focus on a goal and work hard",
  "learning a new skill opens the door"
];

const SWIPE_SENTENCES = [
  "open your mind",
  "enjoy your day",
  "share your smile",
  "love the moment",
  "feel the wind",
  "start a journey",
  "follow your dream",
  "chase the light",
  "seek the truth",
  "find your path",
  "hold the vision",
  "trust your heart",
  "accept the change",
  "build your strength",
  "rise above fear",
  "keep moving forward",
  "stay the course",
  "make a choice",
  "see the beauty",
  "create a memory",
  "show your grace",
  "enjoy the silence",
  "listen to music",
  "live your purpose",
  "honor the promise"
];

const MAIN_SENTENCES = [
  "i see a bright sun in sky",
  "we learn to love and give hope",
  "she reads a book to know more",
  "they play games and make new friends",
  "he writes words to share his mind",
  "we find joy in each new day",
  "i feel calm when music plays",
  "you can grow and learn each time",
  "they walk and talk with good friends",
  "she helps others to grow and learn",
  "you hold hope in your heart today",
  "we want to make life better now",
  "he takes time to watch the sky",
  "they see beauty in small simple things",
  "you make a difference when you care",
  "i learn new things and use them",
  "she gives love and gets love back",
  "we share food and stories with family",
  "he finds peace in music and art",
  "they write words that touch the soul",
  "you show a kind act every day",
  "i play and laugh in bright sunshine",
  "we use our mind to solve problems",
  "he knows that each day brings change",
  "she buys and sells things with care",
  "we meet many new people",
  "i try and try until I succeed",
  "she runs to stay strong",
  "we eat and drink to stay healthy",
  "you rest and take time to grow"
];



// ==== Helper to show/hide sections ====
function showSection(id) {
  document.querySelectorAll("section").forEach(s => s.classList.remove("active"));
  document.getElementById(id).classList.add("active");
}


// ==== Participant ID registration ====
async function registerParticipant(rawID) {
  try {
    let resp = await fetch("/typing/register", {
      method: "POST",
      headers: { "Content-Type":"application/json" },
      body: JSON.stringify({ id: rawID })
    });
    if (resp.ok) {
      let { id } = await resp.json();
      return id;
    }
  } catch(e) { /* ignore */ }
  // fallback to raw
  return rawID;
}


// ==== Initialization: Spacebar to start ====
// document.addEventListener("keydown", async ev => {
//   if (ev.key === "enter" && document.getElementById("init-section").classList.contains("active")) {
//     ev.preventDefault();
//     const input = document.getElementById("participant-id").value.trim();
//     if (!input) return alert("Please enter your ID first.");
//     const uniqueID = await registerParticipant(input);
//     studyState.participantID = uniqueID;
//     showSection("main-menu");
//   }
// }, true);
document.getElementById("btn-start-study")
  .addEventListener("click", async () => {
    const input = document.getElementById("participant-id").value.trim();
    if (!input) return alert("Please enter your ID first.");
    const uniqueID = await registerParticipant(input);
    studyState.participantID = uniqueID;
    showSection("main-menu");
  });

// ==== Main Menu Buttons ====
document.getElementById("btn-normal-practice")
  .addEventListener("click", () => startPhase("normal"));
document.getElementById("btn-swipe-practice")
  .addEventListener("click", () => startPhase("swipe"));
document.getElementById("btn-main-study")
  .addEventListener("click", () => startPhase("main"));

// ==== Start a Phase ====
function shuffle(array) {
  const a = array.slice();            
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function startPhase(phase) {
  studyState.phase = phase;
  studyState.trialIndex = 0;

  switch (phase) {
    case "normal":
      studyState.sentences = shuffle(NORMAL_SENTENCES);
      STUDY_LOGGING_ENABLED = false;
      break;
    case "swipe":
      studyState.sentences = shuffle(SWIPE_SENTENCES);
      STUDY_LOGGING_ENABLED = false;
      break;
    case "main":
      studyState.sentences = shuffle(MAIN_SENTENCES);
      STUDY_LOGGING_ENABLED = true;
      break;
  }

  showSection("task-section");
  loadTrial();
}

function endPhase() {
  // When we’ve finished the MAIN study, show the “Study completed!” screen.
  // Otherwise (after a practice phase) go back to the main menu.
  if (studyState.phase === "main") {
    showSection("completion-section");
  } else {
    showSection("main-menu");
  }
}

// ==== Load the current trial ====
function loadTrial() {
  const total = studyState.sentences.length;
  const current = studyState.trialIndex + 1;
  document.getElementById("trial-counter").textContent =
    `Trial ${current}/${total}  Please type the following sentence`;
  document.getElementById("prompt-box").textContent =
    studyState.sentences[studyState.trialIndex];

  studyState.swipeCount = 0;

  // RESET EVERYTHING
  clearInput();                 // clears formattedInput & currentWord
  typedWords = [];              // <-- actually empty your sentence buffer
  currentTypedWord = "";        // clear last prediction
  display.innerHTML = "";       // clear the display box
  display.style.borderColor = "#ccc";
  predictionBar.replaceChildren(); // remove old buttons
}



// ==== Handle Enter to submit sentence ====
// capture in the _capture_ phase so process.js predict() won't fire
document.addEventListener("keydown", ev => {
  if (ev.key === "Enter" && document.getElementById("task-section").classList.contains("active")) {
    ev.preventDefault();
    ev.stopImmediatePropagation();
    handleSubmitSentence();
  }
}, true);


function handleSubmitSentence() {
  const displayEl = document.getElementById("display");

  // 1) Auto-confirm any in-flight word
  if (formattedInput.length > 0 || currentTypedWord) {
    const last = (currentTypedWord || formattedInput.join("")).toLowerCase().trim();
    if (last) {
      typedWords.push(last);
      typedWords.push(" ");
    }
    clearInput();
  }

  // 2) Build actual vs. expected
  const actual = typedWords.join("").trim().toLowerCase();
  const expected = studyState.sentences[studyState.trialIndex].trim().toLowerCase();
  const wordCount = expected.split(/\s+/).length;

  // 3) Correctness check
  if (actual !== expected) {
    console.log("Expected:", expected, "Got:", actual);
    displayEl.style.borderColor = "red";
    return;
  }

  // 4) Swipe-practice check
  if (studyState.phase === "swipe" && studyState.swipeCount < wordCount) {
    displayEl.style.borderColor = "red";
    alert("Please use a swipe for each word before submitting.");
    return;
  }

  // 5) Pass → green feedback, then advance
  displayEl.style.borderColor = "green";
  setTimeout(() => {
    studyState.trialIndex++;
    if (studyState.trialIndex < studyState.sentences.length) {
      loadTrial();
    } else {
      endPhase();
    }
  }, 500);
}


// ==== Completion: Spacebar to download logs ====
// document.addEventListener("keydown", ev => {
//   if (ev.key === " " && document.getElementById("completion-section").classList.contains("active")) {
//     ev.preventDefault();
//     // trigger your CSV downloads
//     downloadGestureCSV();
//     downloadWordCSV();
//   }
// }, true);
document.getElementById("btn-download-gesture-logs")
  .addEventListener("click", () => {
    downloadGestureCSV();
  });
document.getElementById("btn-download-word-logs")
  .addEventListener("click", () => {
    downloadWordCSV();
  });