My submission for the technical tests of LeWagon Career week. 

Didn't have much time for documentation, sorry !

Check the Makefile to see what the programme can do for you.

(e.g. `make train`, `make api-ping`)

Normally : 
* my docker container should be up and running at `https://technical-test-wfibpwbfra-ew.a.run.app` and the endpoints '/' (GET) and '/predict' (POST) should be available. It is probably not at the moment, since I don't want to let it run without any need for it.
* you should make a .env file if you want to make local tests (e.g. `make api-server` then `make api-predict`)

You can : 
* pass a .JSON file (like test_data.json from this repo) to the prediction endpoint (at /predict). Some columns have been dropped, check them out.
* deploy the package as a container, all the docker information should work, but the GCS (or AWS) part is on you.



