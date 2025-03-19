import React, { useState, useRef, useEffect } from 'react';
import { saveAs } from 'file-saver'; // To save JSON file
import './App.css'; // External CSS file for styles

const App = () => {
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('');
  const [income, setIncome] = useState('');
  const [education, setEducation] = useState('');
  const [empExp, setEmpExp] = useState('');
  const [homeOwnership, setHomeOwnership] = useState('');
  const [loanAmnt, setLoanAmnt] = useState('');
  const [loanIntent, setLoanIntent] = useState('');
  const [loanStatus, setLoanStatus] = useState('');
  const [recording, setRecording] = useState(false);
  const [speechResult, setSpeechResult] = useState('');
  const videoRef = useRef(null);
  const [mediaStream, setMediaStream] = useState(null);

  // Start video capture (WebRTC)
  useEffect(() => {
    if (navigator.mediaDevices) {
      navigator.mediaDevices.getUserMedia({ video: true, audio: true })
        .then(stream => {
          setMediaStream(stream);
          videoRef.current.srcObject = stream;
        })
        .catch(error => console.error('Error accessing webcam:', error));
    }
  }, []);

  // Initialize speech recognition (Web Speech API)
  const startListening = () => {
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = 'en-US';
    recognition.continuous = true;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      console.log('Speech recognition started');
      setRecording(true);
    };

    recognition.onend = () => {
      console.log('Speech recognition ended');
      setRecording(false);
    };

    recognition.onresult = (event) => {
      const transcript = event.results[event.resultIndex][0].transcript;
      console.log('Transcript:', transcript);
      setSpeechResult(transcript);
      handleSpeechInput(transcript);
    };

    recognition.start();
  };

  // Process the speech input (age, income, gender, education, etc.)
  const handleSpeechInput = (transcript) => {
    console.log("Transcript:", transcript);

    // Detect Age
    if (/age\s*is\s*(\d+)/i.test(transcript)) {
      const ageMatch = transcript.match(/age\s*is\s*(\d+)/i);
      if (ageMatch) {
        setAge(ageMatch[1]);
      }
    }

    // Detect Gender
    if (/gender/i.test(transcript)) {
      if (/male|man/i.test(transcript)) {
        setGender('Male');
      } else if (/female|woman/i.test(transcript)) {
        setGender('Female');
      } else {
        setGender('Not Provided');
      }
    }

    // Detect Income
    if (/income\s*is\s*(\d+)/i.test(transcript)) {
      const incomeMatch = transcript.match(/income\s*is\s*(\d+)/i);
      if (incomeMatch) {
        setIncome(incomeMatch[1]);
      } else {
        setIncome('Not Provided');
      }
    }

    // Detect Education
    if (/education\s*is\s*(\w+)/i.test(transcript)) {
      const educationMatch = transcript.match(/education\s*is\s*(\w+)/i);
      if (educationMatch) {
        setEducation(educationMatch[1]);
      }
    }

    // Detect Employment Experience
    if (/experience\s*is\s*(\d+)/i.test(transcript)) {
      const expMatch = transcript.match(/experience\s*is\s*(\d+)/i);
      if (expMatch) {
        setEmpExp(expMatch[1]);
      }
    }

    // Detect Home Ownership
    if (/home\s*ownership/i.test(transcript)) {
      if (/own/i.test(transcript)) {
        setHomeOwnership('Own');
      } else if (/rent/i.test(transcript)) {
        setHomeOwnership('Rent');
      } else if (/mortgage/i.test(transcript)) {
        setHomeOwnership('Mortgage');
      } else {
        setHomeOwnership('Not Provided');
      }
    }

    // Detect Loan Amount
    if (/loan\s*(amount)?\s*(\d+|\w+)/i.test(transcript)) {
      const loanAmntMatch = transcript.match(/loan\s*(amount)?\s*(\d+|\w+)/i);
      if (loanAmntMatch) {
        setLoanAmnt(loanAmntMatch[2]);
      }
    }

    // Detect Loan Intent
    if (/loan\s*intent/i.test(transcript)) {
      if (/home/i.test(transcript)) {
        setLoanIntent('Home');
      } else if (/education/i.test(transcript)) {
        setLoanIntent('Education');
      } else if (/personal/i.test(transcript)) {
        setLoanIntent('Personal');
      } else {
        setLoanIntent('Not Provided');
      }
    }
  };

  // Loan Eligibility Logic (based on age)
  const checkLoanEligibility = async (userAge) => {
    const userData = {
      Age: age || 'Not Provided',
      Gender: gender || 'Not Provided',
      Income: income || 'Not Provided',
      Education: education || 'Not Provided',
      EmploymentExperience: empExp || 'Not Provided',
      HomeOwnership: homeOwnership || 'Not Provided',
      LoanAmount: loanAmnt || 'Not Provided',
      LoanIntent: loanIntent || 'Not Provided',
    };
  
    try {
      const response = await fetch('http://localhost:3000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(userData)
      });
  
      const data = await response.json();
      if (data.status === 'Error') {
        alert('Prediction failed: ' + data.reason);
      } else {
        setLoanStatus(data.prediction || 'No prediction returned');
      }
    } catch (error) {
      console.error('Error calling backend:', error);
      alert('Backend call failed');
    }
  };

  // Save the result to a JSON object file
  const saveToCSV = () => {
    const jsonData = {
      Age: age || 'Not Provided',
      Gender: gender || 'Not Provided',
      Income: income || 'Not Provided',
      Education: education || 'Not Provided',
      EmploymentExperience: empExp || 'Not Provided',
      HomeOwnership: homeOwnership || 'Not Provided',
      LoanAmount: loanAmnt || 'Not Provided',
      LoanIntent: loanIntent || 'Not Provided',
      LoanStatus: loanStatus || 'Not Provided',
    };
    const jsonBlob = new Blob([JSON.stringify(jsonData, null, 2)], { type: 'application/json;charset=utf-8;' });
    saveAs(jsonBlob, 'loan_eligibility_data.json');
  };
  

  return (
    <div className="App">
      <h1>AI Banking Assistant</h1>
      <div className="video-container">
        <video ref={videoRef} autoPlay muted />
      </div>
      <button onClick={startListening} disabled={recording}>
        {recording ? 'Listening...' : 'Start Speaking'}
      </button>
      <div className="results">
        <p><strong>Detected Age:</strong> {age || 'Not Provided'}</p>
        <input type="text" value={age} onChange={(e)=>setAge(e.target.value)}></input>
        <p><strong>Detected Gender:</strong> {gender || 'Not Provided'}</p>
        <input type="text" value={gender} onChange={(e)=>setGender(e.target.value)}></input>
        <p><strong>Detected Income:</strong> {income || 'Not Provided'}</p>
        <input type="text" value={income} onChange={(e)=>setIncome(e.target.value)}></input>
        <p><strong>Detected Education:</strong> {education || 'Not Provided'}</p>
        <input type="text" value={education} onChange={(e)=>setEducation(e.target.value)}></input>
        <p><strong>Employment Experience:</strong> {empExp || 'Not Provided'}</p>
        <input type="text" value={empExp} onChange={(e)=>setEmpExp(e.target.value)}></input>
        <p><strong>Home Ownership:</strong> {homeOwnership || 'Not Provided'}</p>
        <input type="text" value={homeOwnership} onChange={(e)=>setHomeOwnership(e.target.value)}></input>
        <p><strong>Loan Amount:</strong> {loanAmnt || 'Not Provided'}</p>
        <input type="text" value={loanAmnt} onChange={(e)=>setLoanAmnt(e.target.value)}></input>
        <p><strong>Loan Intent:</strong> {loanIntent || 'Not Provided'}</p>
        <input type="text" value={loanIntent} onChange={(e)=>setLoanIntent(e.target.value)}></input>
        <p><strong>Loan Approval</strong>{loanStatus||"Underprocessing"}</p>
        <p><strong>Speech Result:</strong> {speechResult || 'Waiting for input'}</p>
      </div>
      <button onClick={() => checkLoanEligibility(age)}>
  Check Loan Eligibility
</button>
      </div>
  );
};

export default App;