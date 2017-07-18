`Help > Install New Software...`

  * Eclipse Java Development Tools
  * Eclipse Plug-in Development Environment

`File > Import... > Git > Projects from Git > Clone URI`

Enter the URI: https://github.com/lawmurray/Birch.Eclipse.git
Enter username and password
Next, Next again (master should be checked)
Directory: suggest changing .../git/... to .../workspace/...
Next again
Import existing Eclipse projects
Should compile automatically, otherwise `Project > Build Project`.

Now install:
`File > Export... > Plug-in Development > Deployable plug-ins and fragments`
Check Birch (0.0.0)...
Select `Install into host. Repository`

C/C++ > Build > Build Variables in either Project Properties or Eclipse Preferences, may need to add variables like CXX_INCLUDE_PATH and LD_LIBRARY_PATH here.
