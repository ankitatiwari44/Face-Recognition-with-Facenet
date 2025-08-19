import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [attendance, setAttendance] = useState([]);
  const [image, setImage] = useState(null);
  const [status, setStatus] = useState('');

  const fetchAttendance = async () => {
    try {
      const res = await axios.get("http://localhost:5000/get-attendance");
      if (Array.isArray(res.data)) {
        setAttendance(res.data);
      } else {
        console.error("Invalid attendance data format", res.data);
      }
    } catch (err) {
      console.error("Error fetching attendance:", err);
    }
  };

  const startWebcam = async () => {
    try {
      const res = await axios.get("http://localhost:5000/start-recognition");
      setStatus(res.data.status || 'started');
    } catch (err) {
      console.error("Error starting webcam:", err);
    }
  };

  const stopWebcam = async () => {
    try {
      const res = await axios.get("http://localhost:5000/stop-recognition");
      setStatus(res.data.status || 'stopped');
    } catch (err) {
      console.error("Error stopping webcam:", err);
    }
  };

  const handleUpload = async () => {
    if (!image) return alert("Please select an image first.");
    const formData = new FormData();
    formData.append('photo', image);
    try {
      const res = await axios.post('http://localhost:5000/upload-photo', formData);
      alert(res.data.message || 'Photo processed');
      fetchAttendance(); // 🟢 Refresh table after upload
    } catch (err) {
      console.error('Upload failed:', err);
    }
  };

  useEffect(() => {
    fetchAttendance(); // initial load

    // ⏱ Auto-refresh attendance every 10 seconds
    const interval = setInterval(() => {
      fetchAttendance();
    }, 10000);

    return () => clearInterval(interval); // cleanup
  }, []);

  return (
    <div className="App">
      <h1>Face Recognition Attendance</h1>

      <div>
        <button onClick={startWebcam}>Start Webcam</button>&nbsp;
        <button onClick={stopWebcam}>Stop Webcam</button>
        <p>Status: {status}</p>

        <img
          src="http://localhost:5000/video_feed"
          alt="Live Feed"
          width="500"
          height="350"
          style={{ border: "2px solid black", marginTop: "10px" }}
        />

        <div style={{ marginTop: "20px" }}>
          <input type="file" accept="image/*" onChange={(e) => setImage(e.target.files[0])} />
          <button onClick={handleUpload}>Upload & Mark Attendance</button>
        </div>
      </div>

      <h2>Attendance Records</h2>
      <table border="1" cellPadding="10" style={{ margin: "auto" }}>
        <thead>
          <tr>
            <th>Name</th>
            <th>Time</th>
          </tr>
        </thead>
        <tbody>
          {attendance.length === 0 ? (
            <tr>
              <td colSpan="2">No attendance found</td>
            </tr>
          ) : (
            attendance.map((entry, idx) => (
              <tr key={idx}>
                <td>{entry.name}</td>
                <td>{entry.time}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

export default App;
