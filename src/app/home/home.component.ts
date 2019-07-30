import { Component, OnInit } from '@angular/core';
import { HttpHandlerService } from '../services/http-handler.service';
import { FormBuilder, FormGroup } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { AuthService } from '../services/auth.service';
import { VideoUtilService } from '../services/video-util.service';
import { Router } from '@angular/router';
import { environment } from '../../environments/environment'
import { setAData,setLData,setCData } from '../../assets/sharedData'

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent implements OnInit {
  fileStatus: string;
  spinnerStatus: boolean;
  SERVER_URL = "http://"+environment.IP+"/upload_video";
  uploadForm: FormGroup;  
  isVideoSelected: boolean;
  userID : string;
  videos: any = [];
  
  constructor(private formBuilder: FormBuilder,
    private httpClient: HttpClient, private auth: AuthService,
    private videoUtil: VideoUtilService, private httpHandler: HttpHandlerService,
    private router: Router) { 
    this.spinnerStatus = true;
    this.fileStatus = ''
    this.isVideoSelected = true;
    console.log('In home comp constructor');
    
    // this.userID = this.auth.currentUserId;
    if( this.auth == undefined){
      this.userID = localStorage.getItem('userId')
    }
    else if(this.auth !=null){
      if(this.auth.currentUserId != null){
        this.userID = this.auth.currentUserId
      }
    }
    else{
     this.userID = null
     console.log('In user = null');
     
    } 
  }

  ngOnInit() {
    console.log('in inti home cmp');
    
    this.userID = localStorage.getItem('userId')
    this.httpHandler.getUserVideos(this.userID)
    .subscribe(resp =>{this.videos = resp; console.log(resp);
    } );
    
    this.uploadForm = this.formBuilder.group({
      profile: ['']
    });  
  }

  onFileSelect(event) {
    if (event.target.files.length > 0) {
      const file = event.target.files[0];  
      this.uploadForm.get('profile').setValue(file);
      this.isVideoSelected=false;
      this.playVideo(file);
    }
  }

  onSubmit() {

    this.spinnerStatus = false;
    const formData = new FormData();
    const fileObject = this.uploadForm.get('profile').value;
   
    formData.append('file', fileObject);        
    formData.append('userid',this.auth.currentUserId);
    this.httpClient.post<any>(this.SERVER_URL, formData, 
      {responseType: 'json',  reportProgress: true,
      } ).subscribe(
      (res) =>{ 
        this.spinnerStatus = true;
        console.log(res);
        if(res.success === 'true'){
          this.fileStatus = 'File Uploaded'
        }
        else{
          this.fileStatus = 'File Not Uploaded'
        }
      },
      (err) => { 
        console.log(err); 
        this.spinnerStatus = true;
        this.fileStatus = 'Error uploading file'}
    ); 
  }

  playVideo(file){
    const URL = window.URL;  
    const fileURL = URL.createObjectURL(file)
    this.videoUtil.setFileURL(fileURL);
    const videoNode = document.querySelector('video')
    videoNode.src = fileURL
  }
  playVideoByURL(cVideoURL){
    const videoNode = document.querySelector('video')
    videoNode.src = cVideoURL;
  }
  goToAnalysis(objectId,videoURL){
    this.videoUtil.setFileURL(videoURL);
    this.httpHandler.getVideoResults(objectId)
    .subscribe(resp =>{ 
      let localeRes = resp[0].locale_response
      let cadRes = resp[0].angle_response
      let aparallelRes = resp[0].apparel_response 
      setLData(JSON.parse(localeRes))
      setAData(JSON.parse(aparallelRes))
      setCData(JSON.parse(cadRes))
      this.router.navigateByUrl('/analysis')
      });
  }

}
