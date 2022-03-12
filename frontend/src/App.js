
import React, {useRef,useState,useEffect} from 'react'


function App() {
  const [data, setData]=useState([{}])
  const videoRef=useRef(null);
  const photoRef=useRef(null);
  const [hasPhoto,setHasPhoto]=useState(false);
  const getVideo=()=>{
    navigator.mediaDevices
      .getUserMedia({
        video:{width: 1920, height:1080 }
        })
    .then(stream=>{ 
        let video=videoRef.current;
        video.srcObject=stream;
        video.play();
    })
    .catch(err=> {
      console.error(err);
    })
  }

  const takePhoto=()=> {
    const width=414;
    const height=width/(16/9);

    let video=videoRef.current;
    let photo=photoRef.current;

    photo.width=width;
    photo.height=height;

    let ctx=photo.getContext('2d');
    ctx.drawImage(video,0,0,width,height);

    setHasPhoto(true);
  }

  const closePhoto = () => {
    let photo=photoRef.current;
    let ctx=photo.getContext('2d');
    ctx.clearRect(0,0,photo.width,photo.height);
    setHasPhoto(false);
  }

  useEffect(()=>{
    getVideo();
  },[videoRef])


  useEffect(()=> {
    fetch("/prediction").then(
      res=>res.json()
    ).then(
        data=>{
          setData(data)
          console.log(data)
        }
    )
  },[])


  return (
    <div className="App">
      <div className='camera'>
            <video ref={videoRef}></video>
            <button onClick={takePhoto}>Capture</button>
      </div>
      <div className={'result'+(hasPhoto ? 'hasPhoto':'')}>
        <canvas ref={photoRef}></canvas>
        <button onClick={closePhoto}>Click Another Pic</button>
        <p>Product : {data.a}</p>
      </div>
    </div>
   
      // <div>
      // <p>Product : {data.a}</p>
      // </div>
  )
}

export default App;
