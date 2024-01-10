const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path')
const cors = require('cors')

const app = express();
const port = 5000;

app.use(express.static('public'));
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json())
app.use(cors())
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });


app.get('/', (req, res) => res.send("This is the HomePage Test of -|Secure Url Shield-|"))

app.post('/upload', upload.single('file'), (req, res) => {
      try {
        const fileUpload = req.file;
           fs.writeFileSync('uploaded_dataset.csv', fileUpload.buffer)
           console.log(filePath)
              const label = fileUpload.originalname
                console.log(label)
              res.json({message: 'File Uploaded Successfully!', label})

      } catch (error) {
        console.log(error);
      }
});

app.post('/process-url', (req, res) => {

    try {
        
        const filePath = path.join(__dirname, 'uploaded_dataset.csv');
        // const fileName = getFileName(filePath);
        const dataset = path.basename(filePath);
           const url = req.body.url;
               const pythonProcess = spawn('python', ['trained_model.py', url, dataset]);

        // Handle stdout and stderr events
        let label = '';
        pythonProcess.stdout.on('data', (data) => {
          console.log(`Python script output: ${data}`);
          label += data.toString().trim()
          // You can send this data back to the client if needed
        });
       
       
        pythonProcess.stderr.on('data', (data) => {
          console.error(`Python script error: ${data}`);
          // Handle error if needed
        });
        
        // Close event is emitted when the process exits
        pythonProcess.on('close', (code) => {
          console.log(`Python script exited with code ${code}`);
          // Handle the process exit code if needed
          res.json({ message: 'Processing completed successfully', label });
        });

    } catch (error) {
      console.error(error)
    }
})

app.listen(port, () => {
  console.log(`Server listening at http://localhost:${port}`);
});
