import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class VideoUtilService {
 fileURL: any;
  constructor() { }
  setFileURL(url){
    this.fileURL = url;
  }
  getFileURL(){
    return this.fileURL;
  }
}
