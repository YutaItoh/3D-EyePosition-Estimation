#ifndef __TRACKER_LOG_H__
#define __TRACKER_LOG_H__

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include "timer.h"


#define TRACKER_LOG

struct tracker_log
{
public:
	typedef std::vector< std::pair<std::string, std::string> >::iterator iterator;
	typedef std::vector< std::pair<std::string, std::string> >::const_iterator const_iterator;

	template<typename T>
	void add(const std::string& key, const T& val)
	{
		m_log.push_back(std::make_pair(key, boost::lexical_cast<std::string>(val)));
	}
	void add(const std::string& key, const timer& val)
	{
		std::stringstream ss;
		ss.precision(2);
		ss.setf(std::ios::fixed);
		ss << (val.elapsed()*1000.0) << "ms";
		m_log.push_back(std::make_pair(key, ss.str()));
	}

	iterator begin() { return m_log.begin(); }
	const_iterator begin() const { return m_log.begin(); }
	iterator end() { return m_log.end(); }
	const_iterator end() const { return m_log.end(); }
	void print(){
#ifdef TRACKER_LOG
		tracker_log::iterator itr = this->begin();
		const tracker_log::const_iterator kItrEnd = this->end();
		while( itr != kItrEnd ){
			std::cout << "Log: " << itr->first << ": "<< itr->second << std::endl;
			++itr;
		}
#endif
	}
private:
	std::vector< std::pair<std::string, std::string> > m_log;
};

namespace 
{
	struct section_guard
	{
		std::string name;
		tracker_log& log;
		timer t;
		section_guard(const std::string& name, tracker_log& log) : name(name), log(log), t() {  }
		~section_guard() { log.add(name, t); }
		operator bool() const {return false;}
	};

	inline section_guard make_section_guard(const std::string& name, tracker_log& log)
	{
		return section_guard(name,log);
	}
}

#ifdef TRACKER_LOG
///#define SECTION(A,B) {std::cout<<__FILE__<<" "<<__LINE__<<std::endl;}if (const section_guard& _section_guard_ = make_section_guard( A , B )) {} else
#define SECTION(A,B) if (const section_guard& _section_guard_ = make_section_guard( A , B )) {} else
#else
#define SECTION(A,B) {}
#endif

#endif  __TRACKER_LOG_H__