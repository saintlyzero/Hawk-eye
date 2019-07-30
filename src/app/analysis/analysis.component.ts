import { Component, OnInit } from '@angular/core';
import { VgAPI } from 'videogular2/compiled/core';
import { VideoUtilService } from '../services/video-util.service';
import { getFrameClasses } from '../../assets/sharedData'
import { Router } from '@angular/router';

@Component({
  selector: 'app-analysis',
  templateUrl: './analysis.component.html',
  styleUrls: ['./analysis.component.scss'],
})
export class AnalysisComponent implements OnInit {
  api: VgAPI;
  fileURL: any;
  previousTime: number = -1;
  
  locale: any = [];
  cad: any = [];
  aparallel: any = [];

  constructor(
    public videoUtil: VideoUtilService,
    public route: Router) {
    this.fileURL = videoUtil.getFileURL();
  }
  ngOnInit() {
    this.playVideo(this.fileURL)
  }
   playVideo(fileURL) {
    const videoNode = document.querySelector('video')
    videoNode.src = fileURL
  }

  onPlayerReady(api: VgAPI) {
    this.api = api;
    this.api.getDefaultMedia().subscriptions.timeUpdate.subscribe(
      () => {
        let time = Math.floor(this.api.currentTime);
        if (time - this.previousTime >= 1) {
          
          this.emptyAllArrays();
          this.locale = getFrameClasses(this.previousTime+1,'locale')
          this.cad = getFrameClasses(this.previousTime+1,'cad')
          this.aparallel = getFrameClasses(this.previousTime+1,'aparallel')
          this.previousTime = time;
        }
      }
    );
    // Adjust timer when user backtracks pointer
    this.api.getDefaultMedia().subscriptions.seeked.subscribe(
      () => {
        let time = Math.floor(this.api.currentTime);
        this.previousTime = time - 1;
        if (time - this.previousTime >= 1) {
          this.emptyAllArrays();
          this.locale = getFrameClasses(this.previousTime+1,'locale')
          this.cad = getFrameClasses(this.previousTime+1,'cad')
          this.aparallel = getFrameClasses(this.previousTime+1,'aparallel')
        }
      }
    );
  }

  emptyAllArrays(){
    this.locale = [];
    this.cad = [];
    this.aparallel = [];
  }
  generateGraphs(){
    this.route.navigateByUrl('/stats')
  }
}
