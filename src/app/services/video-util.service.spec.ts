import { TestBed } from '@angular/core/testing';

import { VideoUtilService } from './video-util.service';

describe('VideoUtilService', () => {
  beforeEach(() => TestBed.configureTestingModule({}));

  it('should be created', () => {
    const service: VideoUtilService = TestBed.get(VideoUtilService);
    expect(service).toBeTruthy();
  });
});
