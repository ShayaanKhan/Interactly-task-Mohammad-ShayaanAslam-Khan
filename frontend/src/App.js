import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [jobDescription, setJobDescription] = useState('');
  const [responses, setResponses] = useState([]);

  const handleSearch = async () => {
    const result = await axios.post('http://localhost:5000/match', { job_description: jobDescription });
    setResponses(result.data);
  };

  return (
    <div>
      <input
        type="text"
        value={jobDescription}
        onChange={(e) => setJobDescription(e.target.value)}
        placeholder="Enter job description"
      />
      <button onClick={handleSearch}>Search</button>
      <div>
        {responses.map((response, index) => (
          <p key={index}>{response}</p>
        ))}
      </div>
    </div>
  );
}

export default App;
