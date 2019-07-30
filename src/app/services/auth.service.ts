import { Injectable, NgZone } from '@angular/core';
import { AngularFireAuth } from '@angular/fire/auth';
import { Router } from '@angular/router';
import * as firebase from 'firebase/app'


@Injectable()
export class AuthService {


  authState: any = null
  constructor(
    public afAuth: AngularFireAuth, 
    private router: Router,
    public ngZone: NgZone) {
    this.afAuth.authState.subscribe(data => {
      this.authState = data
      if(this.authState)
      localStorage.setItem('userId', this.authState.uid)
    })
  }
  get authenticated(): boolean {
    return this.authState !== null
  }
  get currentUserId(): string {
    return this.authenticated ? this.authState.uid : null
  }
  login() {
    let success: boolean = false;
    this.afAuth.auth.signInWithPopup(
      new firebase.auth.GoogleAuthProvider()
    ).then(res => {
      console.log();
      
      
      this.ngZone.run(() => {
      // console.log('After login ',this.currentUserId);
      // this.saveUserIdInLocalStorage(this.currentUserId)
        this.router.navigate(['home']);
      });
    }
    ).catch(err => {
      console.log("Error in Login")
      console.log(err)
    });
  }
  logout() {
    this.afAuth.auth.signOut();
    localStorage.clear();
    this.router.navigateByUrl('/login');
  }
  saveUserIdInLocalStorage(uid){
    
    console.log('uid in Save method : ',uid);
    localStorage.setItem('user_id', uid);
    console.log('userid retrived from loaclstaorage: ',localStorage.getItem('user_id'));
  }
}
