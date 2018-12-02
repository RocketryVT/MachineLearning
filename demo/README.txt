All these plots came from health-monitoring.py

error-model.png

This is the data model I used for the Support Vector Machine. The nominal
curve is generated using Johnny's MATLAB script using arbitrary motor config
constants. Measurement curves are just the nominal curves with some gaussian
noise. The anomalous curves are that, with a sinusoid slapped on top.

svm-deviation.png

My original intention was to fit the SVM to the healthy measurement curves,
so anything outside of a reasonable limit would be deemed anomalous, and
this frontier would change with time based on the nominal model. SVMs
however are not magic and I haven't gotten this to work. The model
here seems to work pretty well, though. It's just plotting the deviation of
pressure and thrust from their respective nominal curves as functions of
time; the size of the ellipse in this case is roughly proportional to the
covariance of the noise I gave them. Any thrust-pressure measurement outside
the frontier is flagged as anomalous. This can be applied to any number of
variables to produce an n-dimensional ellipsoid envelope of goodness.

local-outlier-factor.png

This is an unrelated demo of Local Outlier Factor I think is cool. I don't
think LOF will be able to reliably detect continuous anomalies (like the
sinusoidal curve in error-model.png) since it only raises the alarm when
measurements are spread relatively far apart. However it's very good at
detecting instantaneous anomalies and filtering out noise. The size of the
circles in this plot can be roughly interpreted as the uncertainty associated
with their measurements. Note that this works even without a model of what the
nominal function looks like.
