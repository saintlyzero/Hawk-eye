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
    this.afAuth.authState.subscribe(data => this.authState = data)
  }
  get authenticated(): boolean {
    return this.authState !== null
  }
  get currentUserId(): string {
    return this.authenticated ? this.authState.uid : null
  }
  //  TODO: Get username function
  login() {
    let success: boolean = false;
    this.afAuth.auth.signInWithPopup(
      new firebase.auth.GoogleAuthProvider()
    ).then(res => {
      this.ngZone.run(() => {
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
    this.router.navigateByUrl('/login');
  }

  // TODO: Remove this
  show() {
    console.log('Auth Login Service Invoked');
  }
}
