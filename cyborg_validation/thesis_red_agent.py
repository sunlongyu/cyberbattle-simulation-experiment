from __future__ import annotations

from ipaddress import IPv4Address
from typing import Dict, Optional, Set

from CybORG.Agents import BaseAgent
from CybORG.Simulator.Actions import (
    DiscoverNetworkServices,
    DiscoverRemoteSystems,
    ExploitRemoteService,
    Impact,
    PrivilegeEscalate,
    Sleep,
)


class ThesisRedAgent(BaseAgent):
    def __init__(self, name: str = "Red", np_random=None):
        super().__init__(name, np_random)
        self.session = 0
        self.scanned_subnets: Set[object] = set()
        self.scanned_ips: Set[IPv4Address] = set()
        self.exploited_ips: Set[IPv4Address] = set()
        self.escalated_hosts: Set[str] = set()
        self.ip_to_hostname: Dict[IPv4Address, str] = {}
        self.pivot_hosts = ("User1", "User2")
        self.enterprise_hosts = ("Prod_Enterprise", "Honey_Enterprise")
        self.operational_hosts = ("Prod_Operational", "Honey_Operational")

    def train(self, results):
        del results

    def set_initial_values(self, action_space, observation):
        del action_space
        self._update_from_observation(observation)

    def end_episode(self):
        self.scanned_subnets = set()
        self.scanned_ips = set()
        self.exploited_ips = set()
        self.escalated_hosts = set()
        self.ip_to_hostname = {}

    def get_action(self, observation, action_space):
        self._update_from_observation(observation)

        subnet = self._first_valid_subnet(action_space, exclude=self.scanned_subnets)
        if subnet is not None and action_space["action"].get(DiscoverRemoteSystems, False):
            self.scanned_subnets.add(subnet)
            return DiscoverRemoteSystems(subnet=subnet, agent=self.name, session=self.session)

        # Progress the killchain in layers: user pivots, then enterprise targets, then operational targets.
        scan_ip = self._first_valid_ip(
            action_space,
            preferred_hostnames=self.pivot_hosts + self.enterprise_hosts + self.operational_hosts,
            exclude=self.scanned_ips,
        )
        if scan_ip is not None and action_space["action"].get(DiscoverNetworkServices, False):
            self.scanned_ips.add(scan_ip)
            return DiscoverNetworkServices(ip_address=scan_ip, agent=self.name, session=self.session)

        escalate_target = self._first_valid_hostname(
            action_space,
            preferred=self.pivot_hosts + self.enterprise_hosts + self.operational_hosts,
            exclude=self.escalated_hosts,
        )
        if escalate_target is not None and action_space["action"].get(PrivilegeEscalate, False):
            self.escalated_hosts.add(escalate_target)
            return PrivilegeEscalate(hostname=escalate_target, agent=self.name, session=self.session)

        exploit_ip = self._first_valid_ip(
            action_space,
            preferred_hostnames=self.pivot_hosts + self.enterprise_hosts + self.operational_hosts,
            exclude=self.exploited_ips,
        )
        if exploit_ip is not None and action_space["action"].get(ExploitRemoteService, False):
            self.exploited_ips.add(exploit_ip)
            return ExploitRemoteService(ip_address=exploit_ip, agent=self.name, session=self.session)

        impact_target = self._first_valid_hostname(action_space, preferred=("Prod_Operational", "Prod_Enterprise"))
        if impact_target is not None and action_space["action"].get(Impact, False):
            return Impact(hostname=impact_target, agent=self.name, session=self.session)

        return Sleep()

    def _update_from_observation(self, observation):
        if not isinstance(observation, dict):
            return
        for host_data in observation.values():
            if not isinstance(host_data, dict):
                continue
            interfaces = host_data.get("Interface", [])
            system_info = host_data.get("System info", {})
            hostname = system_info.get("Hostname")
            for interface in interfaces:
                ip_addr = interface.get("IP Address")
                if hostname is not None and ip_addr is not None:
                    self.ip_to_hostname[ip_addr] = hostname

    def _first_valid_hostname(self, action_space, preferred=(), exclude=None) -> Optional[str]:
        exclude = exclude or set()
        hostname_space = action_space.get("hostname", {})
        for hostname in preferred:
            if hostname in exclude:
                continue
            if hostname_space.get(hostname, False):
                return hostname
        return None

    def _first_valid_ip(self, action_space, preferred_hostnames=(), exclude=None) -> Optional[IPv4Address]:
        exclude = exclude or set()
        ip_space = action_space.get("ip_address", {})
        candidate_ips = [ip for ip, valid in ip_space.items() if valid and ip not in exclude]
        preferred_ips = []
        fallback_ips = []
        for ip_addr in candidate_ips:
            hostname = self.ip_to_hostname.get(ip_addr)
            if hostname in preferred_hostnames:
                preferred_ips.append(ip_addr)
            else:
                fallback_ips.append(ip_addr)
        if preferred_ips:
            return preferred_ips[0]
        if fallback_ips:
            return fallback_ips[0]
        return None

    def _first_valid_subnet(self, action_space, exclude=None):
        exclude = exclude or set()
        subnet_space = action_space.get("subnet", {})
        for subnet, valid in subnet_space.items():
            if valid and subnet not in exclude:
                return subnet
        return None
