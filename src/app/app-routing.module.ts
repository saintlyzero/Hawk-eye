import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { HomeComponent } from './home/home.component';
import { LoginComponent } from './login/login.component';
import { AnalysisComponent } from './analysis/analysis.component';
import { StatsComponent } from './stats/stats.component';

const routes: Routes = [
{ path: 'login', component: LoginComponent },
{ path: 'home', component: HomeComponent },
{ path: 'analysis', component: AnalysisComponent },
{ path: 'stats', component: StatsComponent },
{ path: '**', redirectTo: 'login'}];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
