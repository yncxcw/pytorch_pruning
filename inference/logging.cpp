//Global definition of logging object
#include "include/logging.h"


namespace trtInference {

//TODO(weich) Should these be declared as static ???
Logger gLogger{Logger::Severity::kINFO};
LogStreamConsumer gLogVerbose{LOG_VERBOSE(gLogger)};
LogStreamConsumer gLogInfo{LOG_INFO(gLogger)};
LogStreamConsumer gLogWarning{LOG_WARN(gLogger)};
LogStreamConsumer gLogError{LOG_ERROR(gLogger)};
LogStreamConsumer gLogFatal{LOG_FATAL(gLogger)}; 

}



