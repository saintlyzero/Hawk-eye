import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Video } from '../../videos';
import { environment } from '../../environments/environment'
import 'rxjs/add/operator/map';

@Injectable({
  providedIn: 'root'
})
export class HttpHandlerService {

  IP = environment.IP;
  GET_VIDEOS_URL = "http://"+this.IP+"/get_user_videos/"
  GET_VIDEOS_RESULTS_URL = "http://"+this.IP+"/get_results/"

  videos: Video[];

  constructor(private _httpClient: HttpClient) {}
  show() {
    console.log('In Http Handler..');
  }

  getUserVideos(userid){
    return this._httpClient.get(this.GET_VIDEOS_URL+userid)
  }

  getVideoResults(objectId){
    return this._httpClient.get(this.GET_VIDEOS_RESULTS_URL+objectId)
  }
}
