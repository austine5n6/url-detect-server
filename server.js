const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const bodyParser = require('body-parser');
const cors = require('cors');

const app = express();
const port = 5000;

app.use(express.static('public'));
app.use(bodyParser.urlencoded({ extended: true }));
app.use(cors())

const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

app.post('/upload', upload.single('file'), (req, res) => {

  try {
    const datasetBuffer = req.file;
    console.log(datasetBuffer)
    
    // Call Python script with the dataset
    const pythonProcess = spawn('python', ['test_model.py'], {
      input: datasetBuffer.originalname,
      stdio: ['pipe', 'pipe', 'pipe']
    });

    pythonProcess.stdout.on('data', (data) => {
      const labelResult = data.toString().trim();
      console.log(labelResult)
    });

    pythonProcess.on('close', (code) => {
      console.log(`Python script exited with code ${code}`);
      res.json({ message: 'File uploaded successfully!', urls });
    });
  } catch (error) {
    console.error('Error', error);
    res.status(500).json({ message: 'Internal server Error'})
  }
});

app.listen(port, () => {
  console.log(`Server listening at http://localhost:${port}`);
});
